import typing as T

import jax.numpy as jnp
import numpy as np
import pytest
import torch

import composer.utils.torch as cxt

# ==================================================================================================
# Conversion


def test_jax_to_torch_inverts_torch_to_jax_for_nested_dicts():
  value = {
    'a': torch.arange(6.0).reshape(2, 3),
    'nested': {'b': torch.tensor([1, 2], dtype=torch.int32)},
  }

  restored = T.cast(dict, cxt.jax_to_torch(cxt.torch_to_jax(value)))

  torch.testing.assert_close(restored['a'], value['a'])
  torch.testing.assert_close(restored['nested']['b'], value['nested']['b'])


def test_torch_to_jax_converts_split_views_and_offset_slices():
  full = torch.arange(24, dtype=torch.float32).reshape(4, 6)
  _, split_view = torch.split(full, [2, 4], dim=1)  # Not contiguous.
  offset_slice = torch.arange(10, dtype=torch.float32)[1:5]  # Starts at a 4-byte offset.

  np.testing.assert_array_equal(cxt.torch_to_jax(split_view), split_view.numpy())
  np.testing.assert_array_equal(cxt.torch_to_jax(offset_slice), offset_slice.numpy())


def test_torch_to_jax_rejects_unsupported_values():
  with pytest.raises(NotImplementedError):
    cxt.torch_to_jax([1, 2])  # type: ignore


def test_jax_to_torch_converts_arrays():
  tensor = cxt.jax_to_torch(jnp.arange(3.0))

  torch.testing.assert_close(tensor, torch.arange(3.0))


# ==================================================================================================
# Flattening


def make_params() -> dict[str, torch.Tensor]:
  """Batched parameters: a [batch=3, 2, 2] weight and a [3, 2] bias that require gradients, and a
  [3, 1] constant that does not."""
  return {
    'weight': torch.zeros(3, 2, 2, requires_grad=True),
    'constant': torch.full((3, 1), 7.0),
    'bias': torch.zeros(3, 2, requires_grad=True),
  }


def test_pack_tensors_keeps_batch_dims_and_skips_tensors_without_grad():
  params = make_params()

  packed = cxt.pack_tensors(params, grad_only=True, batch_dims=1)
  all_packed = cxt.pack_tensors(params, batch_dims=1)

  assert packed.shape == (3, 6)
  assert all_packed.shape == (3, 7)


def test_unpack_tensors_inverts_pack_tensors():
  params = make_params()
  vector = torch.arange(18, dtype=torch.float32).reshape(3, 6)

  unpacked = cxt.unpack_tensors(vector, params, grad_only=True)

  assert unpacked is params  # Written in place.
  torch.testing.assert_close(cxt.pack_tensors(unpacked, grad_only=True, batch_dims=1), vector)
  torch.testing.assert_close(params['constant'], torch.full((3, 1), 7.0))


def test_unpack_tensors_writes_only_the_given_batch_slice():
  params = make_params()

  cxt.unpack_tensors(torch.ones(2, 6), params, grad_only=True, batch_idxs=[slice(0, 2)])

  torch.testing.assert_close(params['bias'][:2], torch.ones(2, 2))
  torch.testing.assert_close(params['bias'][2], torch.zeros(2))


def test_unpack_tensors_rejects_wrong_vector_size():
  with pytest.raises(ValueError, match='Incorrect vector size'):
    cxt.unpack_tensors(torch.zeros(3, 9), make_params(), grad_only=True)


def test_unpack_tensors_rejects_wrong_number_of_batch_slices():
  with pytest.raises(ValueError, match='batch_idxs'):
    cxt.unpack_tensors(torch.zeros(3, 6), make_params(), grad_only=True, batch_idxs=[])


def test_flatten_tensors_joins_nested_keys_in_order():
  value = {'b': torch.zeros(1), 'a': {'y': torch.ones(1), 'x': {'z': torch.ones(2)}}}

  flat = cxt.flatten_tensors(value)

  assert list(flat) == ['b', 'a/y', 'a/x/z']


# ==================================================================================================
# Functional


class TiedNetwork(torch.nn.Module):
  """Two linear layers sharing one weight, with a non-`forward` method."""

  def __init__(self):
    super().__init__()
    self.first = torch.nn.Linear(2, 2, bias=False)
    self.second = torch.nn.Linear(2, 2, bias=False)
    self.second.weight = self.first.weight

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.second(self.first(x))

  def first_only(self, x: torch.Tensor) -> torch.Tensor:
    return self.first(x)


def test_functional_method_uses_given_params_and_restores_module_params():
  network = TiedNetwork()
  original = network.first.weight.detach().clone()
  params = {'first.weight': 2 * torch.eye(2)}

  output = cxt.functional_method(network, (params, {}), args=torch.ones(1, 2))

  torch.testing.assert_close(output, torch.full((1, 2), 4.0))  # Both tied layers used 2 * I.
  torch.testing.assert_close(network.first.weight.detach(), original)
  assert network.second.weight is network.first.weight


def test_functional_method_calls_the_given_method():
  network = TiedNetwork()
  params = {'first.weight': 3 * torch.eye(2)}

  output = cxt.functional_method(
    network,
    (params, {}),
    args=(torch.ones(1, 2),),
    method=network.first_only,
  )

  torch.testing.assert_close(output, torch.full((1, 2), 3.0))


def test_functional_method_rejects_names_in_both_params_and_buffers():
  network = TiedNetwork()
  weight = {'first.weight': torch.eye(2)}

  with pytest.raises(ValueError, match='overlapping'):
    cxt.functional_method(network, (weight, weight), args=torch.ones(1, 2))
