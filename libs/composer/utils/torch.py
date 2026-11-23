import collections
import collections.abc
import typing as T

import jax
import jax.dlpack
import torch
import torch.jit
import torch.nn.utils.stateless
import torch.utils.dlpack

# ==================================================================================================
# Conversion

NestedArrayDict = T.Mapping[str, 'jax.Array | NestedArrayDict']
NestedTensorDict = T.Mapping[str, 'torch.Tensor | NestedTensorDict']
FlatTensorDict = T.Mapping[str, torch.Tensor]
FlatArrayDict = T.Mapping[str, jax.Array]
Device = str | torch.device | None


@T.overload
def torch_to_jax(value: torch.Tensor, device: Device = None) -> jax.Array: ...
@T.overload
def torch_to_jax(value: FlatTensorDict, device: Device = None) -> FlatArrayDict: ...
@T.overload
def torch_to_jax(value: NestedTensorDict, device: Device = None) -> NestedArrayDict: ...
@T.overload
def torch_to_jax(
  value: torch.Tensor | NestedTensorDict,
  device: Device = None,
) -> jax.Array | NestedArrayDict: ...
def torch_to_jax(
  value: torch.Tensor | NestedTensorDict,
  device: Device = None,
) -> jax.Array | NestedArrayDict:
  """Converts a Torch tensor (or a nested dict of them) to JAX arrays through DLPack.

  The arrays share memory with the tensors where possible: a tensor is copied only when DLPack
  cannot take it as is (see below).

  Args:
    value: The tensor, or a nested dict of tensors.
    device: Device to move the tensors to first, or `None` to keep their devices.

  Returns:
    The array, or a nested dict of arrays with the same keys.

  Raises:
    NotImplementedError: For other value types.
  """
  match value:
    case torch.Tensor():
      if device:
        value = value.to(device)
      # DLPack only supports compact (or transposed) strides, e.g. rejects the views produced by
      # `torch.split`, so make sure the buffer is compact before crossing into JAX.
      if not value.is_contiguous():
        value = value.contiguous()
      # On CUDA, a "contiguous" view can still start at a non-16-byte-aligned offset into its
      # parent storage (e.g. a slice like `full[..., a:b]`), which XLA's compiled executables
      # reject at dispatch time. `.contiguous()` doesn't catch this since it only checks strides,
      # so clone to a fresh (offset-0) allocation whenever the pointer itself is misaligned.
      if value.is_cuda and value.data_ptr() % 16 != 0:
        value = value.clone()
      return jax.dlpack.from_dlpack(value)
    case collections.abc.Mapping():
      return {k: torch_to_jax(v, device=device) for k, v in value.items()}
    case _:
      raise NotImplementedError(f'Unsupported type: {type(value)}')


@T.overload
def jax_to_torch(value: jax.Array, device: Device = None) -> torch.Tensor: ...
@T.overload
def jax_to_torch(value: FlatArrayDict, device: Device = None) -> FlatTensorDict: ...
@T.overload
def jax_to_torch(value: NestedArrayDict, device: Device = None) -> NestedTensorDict: ...
@T.overload
def jax_to_torch(
  value: jax.Array | NestedArrayDict,
  device: Device = None,
) -> torch.Tensor | NestedTensorDict: ...
def jax_to_torch(
  value: jax.Array | NestedArrayDict,
  device: Device = None,
) -> torch.Tensor | NestedTensorDict:
  """Converts a JAX array (or a nested dict of them) to Torch tensors through DLPack, sharing
  memory.

  Args:
    value: The array, or a nested dict of arrays.
    device: Device to move the tensors to afterwards, or `None` to keep the arrays' devices.

  Returns:
    The tensor, or a nested dict of tensors with the same keys.

  Raises:
    NotImplementedError: For other value types.
  """
  match value:
    case jax.Array():
      tensor = torch.utils.dlpack.from_dlpack(value)
      if device:
        tensor = tensor.to(device)
      return tensor
    case collections.abc.Mapping():
      return {k: jax_to_torch(v, device=device) for k, v in value.items()}
    case _:
      raise NotImplementedError(f'Unsupported type: {type(value)}')


# ==================================================================================================
# Flattening


def pack_tensors(
  tensors: dict[str, torch.Tensor],
  *,
  grad_only: bool = False,
  batch_dims: int = 0,
) -> torch.Tensor:
  """Flattens a dict of tensors (e.g. a module's parameters) into one vector per batch element.

  The tensors must be on one device and share their leading `batch_dims` axes.

  Args:
    tensors: The tensors, in packing order.
    grad_only: Whether to pack only the tensors that require gradients.
    batch_dims: Number of leading batch axes to keep.

  Returns:
    The tensors flattened and concatenated, `[*batch, size]`.
  """
  vectors = []
  for tensor in tensors.values():
    if grad_only and not tensor.requires_grad: continue
    # Keep the first `batch_dims` of shape and flatten the rest.
    vectors.append(tensor.view(*tensor.shape[:batch_dims], -1))
  return torch.cat(vectors, dim=-1)


def unpack_tensors(
  vector: torch.Tensor,
  tensors: dict[str, torch.Tensor],
  *,
  grad_only: bool = False,
  batch_idxs: T.Sequence[slice] | None = None,
) -> dict[str, torch.Tensor]:
  """Writes a packed vector back into the tensors it was packed from (see `pack_tensors`).

  The values are written in place through `.data` (bypassing autograd), into the tensors' leading
  batch axes, which match the vector's.

  Args:
    vector: The packed values, `[*batch, size]`.
    tensors: The tensors to write into, in packing order. Modified in place.
    grad_only: Whether only the tensors that require gradients were packed.
    batch_idxs: Slices of the tensors' batch axes to write, one per batch axis of `vector`.
      Defaults to all of them.

  Returns:
    `tensors`, written.

  Raises:
    ValueError: If `batch_idxs` does not have one slice per batch axis, or `vector`'s size does not
      match the tensors'.
  """
  # The batch slices of the tensors to write.
  batch_dims = len(vector.shape) - 1
  if batch_idxs is None:
    batch_idxs = [slice(None) for _ in range(batch_dims)]  # Overwrite all batch dims.
  elif len(batch_idxs) != batch_dims:
    raise ValueError(
      f'Length of `batch_idxs` ({len(batch_idxs)}) != Number of `vector` batch dims '
      f'({batch_dims}): `vector` = {list(vector.shape)}, `batch_idxs` = {list(batch_idxs)}'
    )
  else:
    batch_idxs = list(batch_idxs)  # Overwrite only specified batch dims indices.

  # Write each tensor's segment of the vector.
  offset = 0
  for tensor in tensors.values():
    if grad_only and not tensor.requires_grad: continue
    size = tensor.shape[batch_dims:].numel()
    tensor_view = tensor[*batch_idxs]
    try:
      tensor_view.data[...] = vector[..., offset:offset + size].view_as(tensor_view).data
    except RuntimeError as e:
      vector_shape = list(vector.shape)[:-1] + [size]
      e.add_note(
        f'Trying to unflatten {vector_shape} flat vector -> {list(tensor_view.shape)} unflat tensor'
      )
      raise e
    offset += size
  if offset != vector.shape[-1]:
    raise ValueError(f'Incorrect vector size to unflatten: {list(vector.shape)}')
  return tensors


def flatten_tensors(value: NestedTensorDict) -> FlatTensorDict:
  """Flattens a nested dict of tensors to one level, joining nested keys with `/` (e.g. `a/b`),
  in order."""
  out = collections.OrderedDict()

  def flatten(prefix: str, node: NestedTensorDict):
    """Adds the tensors of `node` to `out`, with their keys prefixed by `prefix`."""
    for k, v in node.items():
      if isinstance(v, collections.abc.Mapping):
        flatten(f'{prefix}{k}/', v)
      else:
        out[f'{prefix}{k}'] = v

  flatten('', value)
  return out


# ==================================================================================================
# Functional


def functional_method(
  module: torch.nn.Module,
  params_and_buffers: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
  args: T.Any | tuple | None = None,
  kwargs: dict[str, T.Any] | None = None,
  method: T.Callable | None = None,
  *,
  tie_weights: bool = True,
  strict: bool = False,
) -> T.Any:
  """Calls a method of `module` with its parameters and buffers replaced by the given tensors.

  Like `torch.func.functional_call`, but for any method, not only `forward`. The module's own
  parameters and buffers are unchanged afterwards.

  Args:
    module: The module.
    params_and_buffers: The parameters and buffers to use, by name (`named_parameters()` and
      `named_buffers()` names). The two dicts must not share keys.
    args: Positional arguments of the method: a tuple, or a single argument.
    kwargs: Keyword arguments of the method.
    method: The method to call, of `module` or of one of its submodules. Defaults to the module
      itself (`forward`).
    tie_weights: Whether tied parameters take the given tensor of either name (as
      `torch.func.functional_call`).
    strict: Whether the given tensors must cover all of the module's parameters and buffers.

  Returns:
    The method's return value.

  Raises:
    ValueError: If a name is both a parameter and a buffer.
    RuntimeError: If `module` is scripted, traced or `nn.DataParallel`.
  """
  # Combine parameter and buffer dicts, validating no key overlap.
  params, buffers = params_and_buffers
  repeated_keys = set(params.keys()) & set(buffers.keys())
  if repeated_keys:
    raise ValueError(f"Parameter and buffer dicts have overlapping keys: {repeated_keys}")

  # As `torch.func.functional_call`.
  if (
    torch.jit.is_tracing()
    or torch.jit.is_scripting()
    or isinstance(
      module,
      (
        torch.jit.RecursiveScriptModule,
        torch.jit.ScriptModule,
        torch.jit.ScriptFunction,
      ),
    )
  ):
    raise RuntimeError("The stateless API can't be used with Jitted modules")
  if isinstance(module, torch.nn.DataParallel):
    raise RuntimeError("The stateless API can't be used with nn.DataParallel module")
  if kwargs is None:
    kwargs = {}
  if args is None:
    args = ()
  elif not isinstance(args, tuple):
    args = (args,)

  # Call the method with the tensors swapped in. `torch.func.functional_call` only calls `forward`,
  # so this uses its private `_reparametrize_module` directly.
  original_params: dict[str, torch.Tensor] = dict(module.named_parameters())
  original_buffers: dict[str, torch.Tensor] = dict(module.named_buffers())
  with torch.nn.utils.stateless._reparametrize_module(
    module,
    {**params, **buffers},
    tie_weights=tie_weights,
    strict=strict,
  ):
    output = (method or module)(*args, **kwargs)
  # Overwrite module parameters and buffers to ensure they are unchanged. This is necessary due to
  # a bug in `stateless._reparametrize_module` where it doesn't handle shared modules (and their
  # tied weights) correctly.
  accessor = torch.nn.utils.stateless.NamedMemberAccessor(module)
  accessor.set_tensors_dict(original_params)
  accessor.set_tensors_dict(original_buffers)

  return output
