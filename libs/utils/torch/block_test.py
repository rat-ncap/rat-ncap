import gymnasium
import pytest
import torch

from utils.torch import block as utb


def make_box(size: int, labels: tuple[str, ...] | None = None) -> gymnasium.spaces.Box:
  """Returns a 1D box of `size`, with axis `labels` if given."""
  box = gymnasium.spaces.Box(-1., 1., shape=(size,))
  if labels:
    setattr(box, 'labels', [labels])
  return box


def make_space(**sizes: int) -> gymnasium.spaces.Dict:
  """Returns a dict space of unlabeled 1D boxes of the given sizes."""
  return gymnasium.spaces.Dict({key: make_box(size) for key, size in sizes.items()})


def randomize(module: torch.nn.Module) -> None:
  """Sets every parameter of the module to random values."""
  generator = torch.Generator().manual_seed(0)
  with torch.no_grad():
    for param in module.parameters():
      param.copy_(torch.randn(param.shape, generator=generator))


def make_linear(randomized: bool = True) -> utb.BlockLinear:
  """Returns an initialized `BlockLinear` of inputs x (2), z (3) and outputs y (2), w (2), with a
  shared, a negated and a disabled block, and random (else zero) parameters."""
  layer = utb.BlockLinear(
    input_keys=['x', 'z'],
    output_keys=['y', 'w'],
    weight_ids=[['A', 'B'], ['-(A)', 0]],
    bias_ids=['b', '-(b)'],
  )
  layer.initialize(make_space(x=2, z=3), make_space(y=2, w=2))
  if randomized:
    randomize(layer)
  return layer


# ==================================================================================================
# BlockLinear


def test_blocks_share_negate_and_disable_weights():
  layer = make_linear()
  A, B, b = layer.weights['A'], layer.weights['B'], layer.biases['b']

  weight, bias = layer.assemble_parameters()

  assert sorted(layer.weights) == ['A', 'B'] and sorted(layer.biases) == ['b']
  torch.testing.assert_close(
    weight,
    torch.cat([torch.cat([A, B], dim=1), torch.cat([-A, torch.zeros(2, 3)], dim=1)], dim=0),
  )
  torch.testing.assert_close(bias, torch.cat([b, -b]))


def test_forward_is_the_dense_layer_split_by_output_key():
  layer = make_linear()
  inputs = dict(x=torch.tensor([1., 2.]), z=torch.tensor([3., 4., 5.]))

  outputs = layer(inputs)

  weight, bias = layer.assemble_parameters()
  assert bias is not None
  expected = weight @ torch.cat([inputs['x'], inputs['z']]) + bias
  assert list(outputs) == ['y', 'w']
  torch.testing.assert_close(torch.cat([outputs['y'], outputs['w']]), expected)


def test_negated_view_accumulates_its_gradient_into_the_shared_weight():
  layer = make_linear()
  inputs = dict(x=torch.tensor([1., 2.]), z=torch.tensor([3., 4., 5.]))

  outputs = layer(inputs)
  (outputs['y'].sum() + 2 * outputs['w'].sum()).backward()

  grad = layer.weights['A'].grad
  assert grad is not None
  torch.testing.assert_close(grad, (1 - 2) * inputs['x'].expand(2, 2))


def test_state_dict_holds_the_unique_weights_and_reproduces_the_outputs():
  layer, copy = make_linear(), make_linear(randomized=False)
  inputs = dict(x=torch.tensor([1., 2.]), z=torch.tensor([3., 4., 5.]))

  copy.load_state_dict(layer.state_dict())

  assert sorted(layer.state_dict()) == ['biases.b', 'weights.A', 'weights.B']
  for key, value in layer(inputs).items():
    torch.testing.assert_close(copy(inputs)[key], value)


def test_shared_weight_of_inconsistent_shapes_is_rejected():
  layer = utb.BlockLinear(input_keys=['x', 'z'], output_keys=['y'], weight_ids=[['A', 'A']])

  with pytest.raises(ValueError, match='Inconsistent weight shapes'):
    layer.initialize(make_space(x=2, z=3), make_space(y=2))


def test_generated_weight_ids_are_reserved():
  layer = utb.BlockLinear(input_keys=['x', 'z'], output_keys=['y'], weight_ids=[['$weight0', None]])

  with pytest.raises(ValueError, match='generated'):
    layer.initialize(make_space(x=2, z=2), make_space(y=2))


@pytest.mark.parametrize(
  'space, match',
  [
    (make_space(z=2), 'Missing keys'),
    (gymnasium.spaces.Dict(x=gymnasium.spaces.Box(-1., 1., shape=(2, 2))), 'non-1D'),
  ],
  ids=['missing_key', 'not_1d'],
)
def test_inputs_must_be_present_1d_boxes(space, match):
  with pytest.raises(ValueError, match=match):
    utb.get_flat_box_info(space, ['x'])


# ==================================================================================================
# SparseBlockLinear and BlockScaleShift


def test_sparse_blocks_ignore_the_inputs_of_other_blocks():
  layer = utb.SparseBlockLinear(
    input_keys=[('x',), ('z',)],
    output_keys=[('y',), ('w',)],
    weight_ids=[(('A',),), (('B',),)],
  )
  layer.initialize(make_space(x=2, z=2), make_space(y=2, w=2))
  randomize(layer)
  inputs = dict(x=torch.tensor([1., 2.]), z=torch.tensor([3., 4.]))

  outputs = layer(inputs)
  perturbed = layer(dict(inputs, z=torch.tensor([-3., 9.])))

  torch.testing.assert_close(perturbed['y'], outputs['y'])
  assert not torch.equal(perturbed['w'], outputs['w'])


def test_scale_shift_applies_a_shared_scale_and_a_negated_shift():
  layer = utb.BlockScaleShift(
    input_keys=['x', 'z'],
    weight_ids=['s', 's'],
    bias_ids=['b', '-(b)'],
  )
  layer.initialize(make_space(x=2, z=2))
  randomize(layer)
  s, b = layer.weights['s'], layer.biases['b']
  inputs = dict(x=torch.tensor([1., 2.]), z=torch.tensor([3., 4.]))

  outputs = layer(inputs)

  torch.testing.assert_close(outputs['x'], inputs['x'] * s + b)
  torch.testing.assert_close(outputs['z'], inputs['z'] * s - b)


def test_scale_shift_rejects_a_bias_id_per_missing_key():
  with pytest.raises(AssertionError):
    utb.BlockScaleShift(input_keys=['x', 'z'], bias_ids=['b'])


# ==================================================================================================
# block_initializer


def make_labeled_linear(**kwargs) -> utb.BlockLinear:
  """Returns an initialized `BlockLinear` from labeled input x (a, b) to labeled output y (c, d)."""
  layer = utb.BlockLinear(input_keys=['x'], output_keys=['y'], weight_ids=[['W']], **kwargs)
  layer.initialize(
    gymnasium.spaces.Dict(x=make_box(2, ('a', 'b'))),
    gymnasium.spaces.Dict(y=make_box(2, ('c', 'd'))),
  )
  return layer


def test_initializer_writes_the_value_of_a_labeled_input_output_connection():
  layer = make_labeled_linear(on_initialize=utb.block_initializer(dict(W=[('b', 'c', 0.5)])))

  torch.testing.assert_close(layer.weights['W'].detach(), torch.tensor([[0., 0.5], [0., 0.]]))


def test_initializer_rejects_a_pattern_that_matches_no_block():
  with pytest.raises(ValueError, match='Unused globstrings'):
    make_labeled_linear(on_initialize=utb.block_initializer(dict(V=[('b', 'c', 0.5)])))


def test_connection_initializer_receives_a_view_of_the_entry():
  views = []
  initializer = utb.block_initializer(
    dict(W=[('a', 'd', 1., 'record')]),
    initializers=dict(record=views.append),
  )

  layer = make_labeled_linear(on_initialize=initializer)

  assert len(views) == 1 and views[0].shape == (1, 1)
  assert views[0]._base is layer.weights['W']


def test_bias_init_without_biases_is_rejected():
  with pytest.raises(ValueError, match='no biases'):
    make_labeled_linear(on_initialize=utb.block_initializer({}, bias_init=dict(b=[('c', 1.)])))
