import collections
import dataclasses
import itertools
import types
import typing as T

import gymnasium
import torch

import utils.collections as utc


# ==================================================================================================
# Types

WeightId = str  # Identifier of a weight; blocks of equal IDs share it.
BiasId = str  # Identifier of a bias, as `WeightId`.
DimLabels = T.Sequence[frozenset[str]]  # All string aliases per index along a dimension.

WeightTransform = T.Literal['negate']  # A view of a shared weight (see `WEIGHT_TRANSFORMS`).


@dataclasses.dataclass
class WeightTransformSpec:
  """How a weight transform is written around a weight ID (`<prefix>(<id>)<suffix>`) and the view
  of the weight it gives."""
  prefix: str
  suffix: str
  apply: T.Callable[[torch.nn.Parameter | torch.Tensor], torch.Tensor]


# The weight transforms by name: `'-(id)'` is the negated weight `id`.
WEIGHT_TRANSFORMS = types.MappingProxyType(dict(
  negate=WeightTransformSpec(
    prefix='-',
    suffix='',
    apply=lambda x: -x,
  ),
))

# Characters that a weight ID cannot start or end with (they write transforms). IDs starting with
# `$` are reserved for generated unique IDs.
WEIGHT_ID_RESERVED_CHARACTERS = set('()-')


# ==================================================================================================
# Shared weight modules

class BlockScaleShift(torch.nn.Module):
  """
  A set of parallel element-wise scalings that operate on entries of a mapping. Like
  `BlockLinear`, but each block's weight is a 1D scale vector applied element-wise rather than a
  2D weight matrix applied via matrix multiplication; the corresponding input and output boxes for
  each block must therefore have matching (1D) shape.

  As in `BlockLinear`, each key is assigned a string identifier: keys with the same identifier
  share a scale (and, independently, a shift), `'-(id)'` shares a negated view, `None` is a unique
  (unshared) parameter, and `0` disables it (a constant zero). Biases draw from the same identifier
  namespace as the scales.

  Example:
  ```
  block_scale_shift = BlockScaleShift(
    input_keys=['in1', 'in2', 'in3'],
    weight_ids=['scale', 'scale', None],
    bias_ids=['shift', '-(shift)', 0],
  )
  block_scale_shift.initialize(
    input_space=gymnasium.spaces.Dict(
      in1=gymnasium.spaces.Box(shape=(4,)),
      in2=gymnasium.spaces.Box(shape=(4,)),
      in3=gymnasium.spaces.Box(shape=(6,)),
    )
  )
  print(block_scale_shift)
  print('weights:', block_scale_shift.weights)
  print('biases:', block_scale_shift.biases)
  ```
  ```
  '''
  BlockScaleShift(
    KEY    | in1 (4) | in2 (4)  | in3 (6)
    -------------------------------------
    WEIGHT | scale   | scale    | $scale0
    BIAS   | shift   | -(shift) | 0
  )
  weights: ParameterDict(
    ($scale0): Parameter containing: [torch.FloatTensor of size 6]
    (scale): Parameter containing: [torch.FloatTensor of size 4]
  )
  biases: ParameterDict(
    (shift): Parameter containing: [torch.FloatTensor of size 4]
  )
  '''
  ```
  `in1`/`in2` share the `'scale'` weight and a `'shift'` bias that is negated for `in2`; `in3` gets
  a unique scale (`$scale0`) and no shift (`0`).
  """
  def __init__(
    self,
    input_keys: T.Sequence[str],
    weight_ids: T.Sequence[WeightId | T.Literal[0] | None] | None = None,
    bias_ids: T.Sequence[BiasId | T.Literal[0] | None] | bool = True,
    on_initialize: T.Callable[['BlockScaleShift'], None] | None = None,
  ):
    """
    Args:
      input_keys: The `input_space` keys corresponding to the values to scale and shift.
      weight_ids: One identifier per key in `input_keys`. Equal identifiers share a scale, `'-(id)'`
        shares a negated view, `None` is a unique (unshared) scale, and `0` disables scaling for
        that key (a constant zero). If `None`, every scale is unique.
      bias_ids: One identifier per key in `input_keys`, with the same semantics as `weight_ids`. If
        `False`, no key has a bias. If `True`, every bias is unique.
      on_initialize: A callable that is invoked after initialization.
    """
    super().__init__()
    # Unshared weights.
    if weight_ids is None:
      weight_ids = [None] * len(input_keys)
    if bias_ids is True:
      bias_ids = [None] * len(input_keys)

    # Shape validation.
    assert len(input_keys) == len(set(input_keys)), f'Duplicate input keys: {input_keys}.'
    assert len(input_keys) == len(weight_ids)
    assert bias_ids is False or len(bias_ids) == len(input_keys)

    self.keys = input_keys
    self._weight_ids: T.Sequence[WeightId | T.Literal[0] | None] = weight_ids
    self._bias_ids: T.Sequence[BiasId | T.Literal[0] | None] | T.Literal[False] = bias_ids
    self.on_initialize = on_initialize
    self.initialized = False

  def initialize(self, input_space: gymnasium.spaces.Dict):
    """Creates the scales and shifts for the sizes and labels of the (1D) boxes of `input_space`,
    then calls `on_initialize`. Must be called once before `forward`: the parameters, and so the
    `state_dict`, exist only after it."""
    # Extract the size and labels from each box.
    self.input_sizes, self.input_labels = get_flat_box_info(input_space, self.keys)
    # One `(size,)` shape per key, parallel to weight_ids; broadcasts disabled (`0`) scales.
    self.weight_shapes = list(zip(self.input_sizes))

    # Create ParameterDicts for the shared weights and biases.
    self.weight_ids, self.weights = initialize_shared_weights(
      self._weight_ids,
      self.weight_shapes,
      unique_weight_namespace='scale',
    )
    # Find and aggregate all labels per unique weight ID.
    self.weight_labels = collate_labels(
      self.weight_ids, self.weight_shapes, list(zip(self.input_labels)),
    )

    # `bias_ids` is the single guard for bias-related state: when biases are disabled, none of
    # `biases`/`bias_shapes`/`bias_labels` are set, so every read of them must be gated on it.
    if self._bias_ids is False:
      self.bias_ids: T.Sequence[BiasId | T.Literal[0]] | T.Literal[False] = False
    else:
      # Biases share the same block structure (IDs, shapes, labels) as the scale weights.
      self.bias_shapes = self.weight_shapes
      self.bias_ids, self.biases = initialize_shared_weights(
        self._bias_ids,
        self.bias_shapes,
        unique_weight_namespace='bias',
      )
      # Find and aggregate all labels per unique bias ID.
      self.bias_labels = collate_labels(
        self.bias_ids, self.bias_shapes, list(zip(self.input_labels)),
      )

    # Call the user-provided initialization function.
    if self.on_initialize:
      self.on_initialize(self)

    self.initialized = True

  def forward(self, inputs: T.Mapping[str, torch.Tensor]) -> T.OrderedDict[str, torch.Tensor]:
    """Returns each input times its scale plus its shift, `{key: value}`. The inputs broadcast
    against the 1D scales."""
    outputs: T.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
    for i, key in enumerate(self.keys):
      scale = get_parameter(self.weights, self.weight_ids[i]).broadcast_to(self.weight_shapes[i])
      output = inputs[key] * scale
      if self.bias_ids:
        bias_id = self.bias_ids[i]
        shift = get_parameter(self.biases, bias_id).broadcast_to(self.bias_shapes[i])
        output = output + shift
      outputs[key] = output
    return outputs

  def extra_repr(self) -> str:
    # Transposed relative to BlockLinear's table: keys run across columns (there are typically many
    # of them), and rows are the Weight/Bias identifiers -- biases share the weight identifiers.
    if self.initialized:
      key_headers = [f'{k} ({s})' for k, s in zip(self.keys, self.input_sizes)]
      weight_ids, bias_ids = self.weight_ids, self.bias_ids
    else:
      key_headers = list(self.keys)
      weight_ids, bias_ids = self._weight_ids, self._bias_ids
    table = [['KEY'] + key_headers, ['WEIGHT'] + list(weight_ids)]
    if bias_ids:
      table.append(['BIAS'] + list(bias_ids))
    return format_block_table(table)

  def __repr__(self) -> str:
    return format_module_repr(self)


class BlockLinear(torch.nn.Module):
  """
  A linear layer that subdivides its weight matrix into an uneven grid of blocks based on the
  input and output spaces. `c` input keys correspond to `c` columns, and `r` output keys
  correspond to `r` rows. The size of each block is determined by the length of the boxes
  corresponding to the input and output keys.

  The user has the ability to assign a string identifier to each block, where blocks with the same
  identifier share the same weights. The same applies to biases. The string identifiers can also
  represent transforms on shared weights, e.g. '-(A)' indicates that the block should use a negated
  view of the shared weight identified by 'A'. A `None` identifier is a unique (unshared) block,
  and a `0` identifier disables the block, which then contributes a constant zero.

  Example:
  ```
  block_linear = BlockLinear(
    input_keys=['in1', 'in2'],
    output_keys=['out1', 'out2', 'out3'],
    weight_ids=[
      ['in1out1', 'hello'],
      ['in1out2', '-(in1out2)'],
      ['in2out3', 0],
    ],
    bias_ids=['bias1', '-(bias1)', 'bias3'],
  )
  block_linear.initialize(
    input_space=gymnasium.spaces.Dict(
      in1=gymnasium.spaces.Box(shape=(5,)),
      in2=gymnasium.spaces.Box(shape=(5,)),
    ),
    output_space=gymnasium.spaces.Dict(
      out1=gymnasium.spaces.Box(shape=(3,)),
      out2=gymnasium.spaces.Box(shape=(3,)),
      out3=gymnasium.spaces.Box(shape=(4,))
    )
  )
  print(block_linear)
  print('weights:', block_linear.weights)
  print('biases:', block_linear.biases)
  ```
  ```
  '''
  BlockLinear(
             | BIAS     | in1 (5) | in2 (5)
    ------------------------------------------
    out1 (3) | bias1    | in1out1 | hello
    out2 (3) | -(bias1) | in1out2 | -(in1out2)
    out3 (4) | bias3    | in2out3 | 0
  )
  weights: ParameterDict(
    (hello): Parameter containing: [torch.FloatTensor of size 3x5]
    (in1out1): Parameter containing: [torch.FloatTensor of size 3x5]
    (in1out2): Parameter containing: [torch.FloatTensor of size 3x5]
    (in2out3): Parameter containing: [torch.FloatTensor of size 4x5]
  )
  biases: ParameterDict(
    (bias1): Parameter containing: [torch.FloatTensor of size 3]
    (bias3): Parameter containing: [torch.FloatTensor of size 4]
  )
  '''
  ```
  """
  def __init__(
    self,
    input_keys: T.Sequence[str],
    output_keys: T.Sequence[str],
    weight_ids: T.Sequence[T.Sequence[WeightId | T.Literal[0] | None]] | None = None,
    bias_ids: T.Sequence[BiasId | T.Literal[0] | None] | bool = False,
    on_initialize: T.Callable[['BlockLinear'], None] | None = None,
  ):
    """
    Initializes the BlockLinear layer.

    Args:
      input_keys: A list of input keys corresponding to the columns of the weight matrix.
      output_keys: A list of output keys corresponding to the rows of the weight matrix.
      weight_ids: An `r` x `c` list of identifiers for the weight-matrix blocks. Equal identifiers
        share a weight, `'-(id)'` shares a negated view, `None` is a unique (unshared) block, and
        `0` disables the block (a constant zero). If `None`, every block is unique.
      bias_ids: A list of `r` identifiers for the bias-vector blocks, with the same semantics. If
        `False`, no block has a bias. If `True`, every block's bias is unique.
      on_initialize: A callable that is invoked after the layer is initialized.
    """
    super().__init__()
    # Unshared weights.
    if weight_ids is None:
      weight_ids = [[None] * len(input_keys) for _ in range(len(output_keys))]
    if bias_ids is True:
      bias_ids = [None] * len(output_keys)

    # Shape validation.
    assert len(input_keys) == len(set(input_keys)), f'Duplicate input keys: {input_keys}.'
    assert len(output_keys) == len(set(output_keys)), f'Duplicate output keys: {output_keys}.'
    assert len(output_keys) == len(weight_ids)
    assert all(len(w) == len(input_keys) for w in weight_ids)
    assert bias_ids is False or len(bias_ids) == len(output_keys)

    self.input_keys = input_keys
    self.output_keys = output_keys
    self._weight_ids: T.Sequence[T.Sequence[WeightId | T.Literal[0] | None]] = weight_ids
    self._bias_ids: T.Sequence[BiasId | T.Literal[0] | None] | T.Literal[False] = bias_ids
    self.on_initialize = on_initialize
    self.initialized = False

  def initialize(self, input_space: gymnasium.spaces.Dict, output_space: gymnasium.spaces.Dict):
    """Creates the weights and biases for the sizes and labels of the (1D) boxes of `input_space`
    and `output_space`, then calls `on_initialize`. Must be called once before `forward`: the
    parameters, and so the `state_dict`, exist only after it."""
    # Extract the size and labels from each box.
    self.input_sizes, self.input_labels = get_flat_box_info(input_space, self.input_keys)
    self.output_sizes, self.output_labels = get_flat_box_info(output_space, self.output_keys)

    # Create parallel lists of weight IDs, shapes, and labels for each block.
    weight_ids: list[WeightId | T.Literal[0] | None] = [w for row in self._weight_ids for w in row]
    labels = [(o, i) for o in self.output_labels for i in self.input_labels]
    shapes = [(o, i) for o in self.output_sizes for i in self.input_sizes]

    # Create ParameterDicts for the shared weights and biases.
    resolved_weight_ids, self.weights = initialize_shared_weights(
      weight_ids,
      shapes,
      unique_weight_namespace='weight',
    )
    # Regroup the flat resolved IDs and shapes back into the `r` x `c` block grid. `weight_shapes`
    # is parallel to `weight_ids` and used to broadcast disabled (`0`) blocks during assembly.
    ncols = len(self.input_keys)
    self.weight_ids = [
      resolved_weight_ids[i:i + ncols] for i in range(0, len(resolved_weight_ids), ncols)
    ]
    self.weight_shapes = [shapes[i:i + ncols] for i in range(0, len(shapes), ncols)]
    # Find and aggregate all labels per unique weight ID.
    self.weight_labels = collate_labels(resolved_weight_ids, shapes, labels)

    # Initialize the bias parameters. `bias_ids` is the single guard for bias-related state: when
    # biases are disabled, none of `biases`/`bias_shapes`/`bias_labels` are set, so every read of
    # them must be gated on it. `bias_shapes` is parallel to `bias_ids` (one `(size,)` per output
    # key).
    if self._bias_ids is False:
      self.bias_ids: T.Sequence[BiasId | T.Literal[0]] | T.Literal[False] = False
    else:
      self.bias_shapes = list(zip(self.output_sizes))
      self.bias_ids, self.biases = initialize_shared_weights(
        self._bias_ids,
        self.bias_shapes,
        unique_weight_namespace='bias',
      )
      self.bias_labels = collate_labels(
        self.bias_ids, self.bias_shapes, list(zip(self.output_labels)),
      )

    # Call the user-provided initialization function.
    if self.on_initialize:
      self.on_initialize(self)

    self.initialized = True

  def forward(self, inputs: T.Mapping[str, torch.Tensor]) -> T.OrderedDict[str, torch.Tensor]:
    """Returns the outputs, `{output_key: value}`, of the inputs concatenated in `input_keys` order
    through the assembled weight and bias. The inputs are unbatched 1D tensors: batch with
    `torch.func.vmap`."""
    weight, bias = self.assemble_parameters()

    # Concatenate the input tensors and compute the output tensor.
    input_tensor = torch.cat([inputs[k] for k in self.input_keys], dim=0)
    output_tensor = input_tensor @ weight.T
    if bias is not None:
      output_tensor = output_tensor + bias

    # Split the output tensor into a dictionary of output tensors.
    outputs: T.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
    start = 0
    for i, output_key in enumerate(self.output_keys):
      end = start + self.output_sizes[i]
      outputs[output_key] = output_tensor[start:end]
      start = end
    return outputs

  def assemble_parameters(self) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Assembles the shared weight and bias parameters into a weight matrix and a bias vector.

    Returns:
      A tuple of (weight_tensor, bias_tensor).
    """
    weight = torch.cat([
      torch.cat([
        get_parameter(self.weights, id).broadcast_to(shape)
        for id, shape in zip(id_row, shape_row)
      ], dim=1)
      for id_row, shape_row in zip(self.weight_ids, self.weight_shapes)
    ], dim=0)
    bias = torch.cat([
      get_parameter(self.biases, id).broadcast_to(shape)
      for id, shape in zip(self.bias_ids, self.bias_shapes)
    ], dim=0) if self.bias_ids else None
    return weight, bias

  def resolve_key(
    self,
    key: str | tuple[str, *tuple[int | str | T.Sequence[int | str] | None, ...]],
  ) -> tuple[str, *tuple[list[int], ...]]:
    if isinstance(key, str):
      weight_id, output_index, input_index = key, None, None
    elif len(key) == 2:
      (weight_id, output_index), input_index = key, None
    elif len(key) == 3:
      weight_id, output_index, input_index = key
    else:
      raise IndexError(f'Invalid key: {key}. Expected str or tuple of length 2 or 3.')
    output_indices, input_indices = resolve_indices(
      indices=(output_index, input_index),
      labels=self.weight_labels[weight_id],
    )
    return weight_id, output_indices, input_indices

  def __getitem__(
    self,
    key: str | tuple[str, *tuple[int | str | T.Sequence[int | str] | None, ...]],
  ) -> torch.Tensor:
    weight_id, output_index, input_index = self.resolve_key(key)
    output_index, input_index = broadcast_indices((output_index, input_index))
    return self.weights[weight_id][output_index, input_index]

  def __setitem__(
    self,
    key: str | tuple[str, *tuple[int | str | T.Sequence[int | str] | None, ...]],
    value: torch.Tensor,
  ) -> None:
    weight_id, output_index, input_index = self.resolve_key(key)
    output_index, input_index = broadcast_indices((output_index, input_index))
    self.weights[weight_id].data[output_index, input_index] = value

  def extra_repr(self) -> str:
    if self.initialized:
      input_headers = [f'{k} ({s})' for k, s in zip(self.input_keys, self.input_sizes)]
      output_headers = [f'{k} ({s})' for k, s in zip(self.output_keys, self.output_sizes)]
      weight_ids, bias_ids = self.weight_ids, self.bias_ids
    else:
      input_headers = list(self.input_keys)
      output_headers = list(self.output_keys)
      weight_ids, bias_ids = self._weight_ids, self._bias_ids
    # Output keys label the rows, followed by their biases, then one column per input key.
    table = [[''] + input_headers] + [
      [output_key] + list(row) for output_key, row in zip(output_headers, weight_ids)
    ]
    if bias_ids:
      table = [[row[0], bias] + row[1:] for row, bias in zip(table, ['BIAS'] + list(bias_ids))]
    return format_block_table(table)

  def __repr__(self) -> str:
    return format_module_repr(self)


class SparseBlockLinear(torch.nn.Module):
  """
  A list of independent ("parallel") dense linear blocks. Each block maps a group of input keys
  (concatenated) to a group of output keys (concatenated); unlike `BlockLinear`, whose input and
  output keys index a single dense grid, every block here has its own private input/output keys, so
  the overall structure is block-diagonal.

  Each block is itself a miniature `BlockLinear`: its weight matrix is subdivided into an
  `output_group` x `input_group` grid of sub-blocks, and sub-blocks (within or across blocks) may
  share a weight - or a transformed view of one, e.g. `'-(A)'` - via a common identifier. A
  `None` identifier is a unique (unshared) sub-block, and a `0` identifier disables it (a constant
  zero).

  Example:
  ```
  sparse_block_linear = SparseBlockLinear(
    input_keys=[('in1', 'in2'), ('in3', 'in4')],
    output_keys=[('out1', 'out2'), ('out3', 'out4')],
    weight_ids=[
      [['A', 'B'], ['C', 'D']],
      [['A', '-(B)'], [0, 'D']],
    ],
    bias_ids=[['bias1', 'bias2'], ['bias1', 'bias2']],
  )
  sparse_block_linear.initialize(
    input_space=gymnasium.spaces.Dict(
      in1=gymnasium.spaces.Box(shape=(2,)),
      in2=gymnasium.spaces.Box(shape=(3,)),
      in3=gymnasium.spaces.Box(shape=(2,)),
      in4=gymnasium.spaces.Box(shape=(3,)),
    ),
    output_space=gymnasium.spaces.Dict(
      out1=gymnasium.spaces.Box(shape=(2,)),
      out2=gymnasium.spaces.Box(shape=(3,)),
      out3=gymnasium.spaces.Box(shape=(2,)),
      out4=gymnasium.spaces.Box(shape=(3,)),
    ),
  )
  print(sparse_block_linear)
  print('weights:', sparse_block_linear.weights)
  print('biases:', sparse_block_linear.biases)
  ```
  ```
  '''
  SparseBlockLinear(
             | BIAS  | in1 (2) | in2 (3)
    ------------------------------------
    out1 (2) | bias1 | A       | B
    out2 (3) | bias2 | C       | D

             | BIAS  | in3 (2) | in4 (3)
    ------------------------------------
    out3 (2) | bias1 | A       | -(B)
    out4 (3) | bias2 | 0       | D
  )
  weights: ParameterDict(
    (A): Parameter containing: [torch.FloatTensor of size 2x2]
    (B): Parameter containing: [torch.FloatTensor of size 2x3]
    (C): Parameter containing: [torch.FloatTensor of size 3x2]
    (D): Parameter containing: [torch.FloatTensor of size 3x3]
  )
  biases: ParameterDict(
    (bias1): Parameter containing: [torch.FloatTensor of size 2]
    (bias2): Parameter containing: [torch.FloatTensor of size 3]
  )
  '''
  ```
  The two blocks run independently (block-diagonal), but reuse the same sub-block weights: the
  second block shares the first's `'A'` and `'D'`, takes a negated view of `'B'`, disables its
  lower-left sub-block (`0`), and shares the first block's per-output biases.
  """
  def __init__(
    self,
    input_keys: T.Sequence[tuple[str, ...]],
    output_keys: T.Sequence[tuple[str, ...]],
    weight_ids: T.Sequence[tuple[tuple[WeightId | T.Literal[0] | None, ...], ...]] | None = None,
    bias_ids: T.Sequence[tuple[BiasId | T.Literal[0] | None, ...]] | bool = True,
    on_initialize: T.Callable[['SparseBlockLinear'], None] | None = None,
  ):
    """
    Args:
      input_keys: A list of groups of input keys, one group per block; a group's keys are
        concatenated to form that block's input. The same input key may appear in multiple groups.
      output_keys: A list of groups of output keys, one group per block, parallel to `input_keys`.
        Every output key must appear in exactly one group across the entire list.
      weight_ids: A list of 2D lists, one per block, parallel to `input_keys`/`output_keys`. Each
        block's 2D list is a `len(output_group)` x `len(input_group)` grid of sub-block
        identifiers, with the sharing/`'-(id)'`/`None`/`0` semantics described above. If `None`,
        every sub-block is unique.
      bias_ids: A list of lists, one per block, parallel to `output_keys`; each holds one identifier
        per output key, with the same semantics as `weight_ids`. If `False`, no block has a bias.
        If `True`, every block's biases are unique.
      on_initialize: A callable that is invoked after the layer is initialized.
    """
    super().__init__()
    # Unshared weights.
    if weight_ids is None:
      weight_ids = [
        tuple(tuple(None for _ in in_group) for _ in out_group)
        for in_group, out_group in zip(input_keys, output_keys)
      ]
    if bias_ids is True:
      bias_ids = [tuple(None for _ in out_group) for out_group in output_keys]

    # Parallel input/output key group validation.
    for group in input_keys:
      assert len(group) == len(set(group)), f'Duplicate input keys within a group: {group}.'
    for group in output_keys:
      assert len(group) == len(set(group)), f'Duplicate output keys within a group: {group}.'
    all_output_keys = [k for group in output_keys for k in group]
    assert len(all_output_keys) == len(set(all_output_keys)), (
      f'Every output key must appear in exactly one group; got: {all_output_keys}.'
    )

    # Shape validation.
    assert len(input_keys) == len(output_keys) == len(weight_ids), (
      f'input_keys ({len(input_keys)}), output_keys ({len(output_keys)}), and weight_ids '
      f'({len(weight_ids)}) must all have the same length.'
    )
    assert bias_ids is False or len(bias_ids) == len(weight_ids), (
      f'bias_ids ({len(bias_ids) if bias_ids is not None else 0}) must match the number of '
      f'blocks ({len(weight_ids)}).'
    )
    assert all(
      len(weight_group) == len(output_group) and
      all(len(row) == len(input_group) for row in weight_group)
      for input_group, output_group, weight_group in zip(input_keys, output_keys, weight_ids)
    ), f'Weight IDs must match the shape of the input/output key groups: {weight_ids}.'
    assert bias_ids is False or all(
      len(bias_group) == len(output_group)
      for output_group, bias_group in zip(output_keys, bias_ids)
    ), f'Bias IDs must match the shape of the output key groups: {bias_ids}.'

    self.input_keys = input_keys
    self.output_keys = output_keys
    self._weight_ids: T.Sequence[
      T.Sequence[T.Sequence[WeightId | T.Literal[0] | None]]
    ] = weight_ids
    self._bias_ids: T.Sequence[
      T.Sequence[BiasId | T.Literal[0] | None]
    ] | T.Literal[False] = bias_ids
    self.on_initialize = on_initialize
    self.initialized = False

  def initialize(self, input_space: gymnasium.spaces.Dict, output_space: gymnasium.spaces.Dict):
    """Creates the weights and biases for the sizes and labels of the (1D) boxes of `input_space`
    and `output_space`, then calls `on_initialize`. Must be called once before `forward`: the
    parameters, and so the `state_dict`, exist only after it."""
    # Extract the size and labels from each individual key's box.
    flat_input_keys = sorted(set(k for group in self.input_keys for k in group))
    flat_output_keys = [k for group in self.output_keys for k in group]
    input_sizes, input_labels = get_flat_box_info(input_space, flat_input_keys)
    output_sizes, output_labels = get_flat_box_info(output_space, flat_output_keys)
    input_size_by_key = dict(zip(flat_input_keys, input_sizes))
    input_labels_by_key = dict(zip(flat_input_keys, input_labels))
    output_size_by_key = dict(zip(flat_output_keys, output_sizes))
    output_labels_by_key = dict(zip(flat_output_keys, output_labels))

    # Per-block, per-key sizes, parallel to input_keys/output_keys -- used for splitting `forward`'s
    # concatenated output and for annotating sizes in `extra_repr`.
    self.input_sizes = [[input_size_by_key[k] for k in group] for group in self.input_keys]
    self.output_sizes = [[output_size_by_key[k] for k in group] for group in self.output_keys]

    # Flatten every block's `output_group` x `input_group` sub-block grid into parallel lists of
    # weight IDs, shapes, and labels (block-major, then row-major), mirroring `BlockLinear`.
    weight_ids: list[WeightId | T.Literal[0] | None] = [
      w for block in self._weight_ids for row in block for w in row
    ]
    shapes = [
      (output_size_by_key[o], input_size_by_key[i])
      for in_group, out_group in zip(self.input_keys, self.output_keys)
      for o in out_group for i in in_group
    ]
    labels = [
      (output_labels_by_key[o], input_labels_by_key[i])
      for in_group, out_group in zip(self.input_keys, self.output_keys)
      for o in out_group for i in in_group
    ]

    # Create the shared weight ParameterDict, then regroup the flat resolved IDs and shapes back
    # into each block's sub-block grid. `weight_shapes` is parallel to `weight_ids` and used to
    # broadcast disabled (`0`) sub-blocks during assembly.
    resolved_weight_ids, self.weights = initialize_shared_weights(
      weight_ids,
      shapes,
      unique_weight_namespace='weight',
    )
    self.weight_ids = []
    self.weight_shapes = []
    cursor = 0
    for in_group, out_group in zip(self.input_keys, self.output_keys):
      id_block, shape_block = [], []
      for _ in out_group:
        id_block.append(list(resolved_weight_ids[cursor:cursor + len(in_group)]))
        shape_block.append(list(shapes[cursor:cursor + len(in_group)]))
        cursor += len(in_group)
      self.weight_ids.append(id_block)
      self.weight_shapes.append(shape_block)
    # Find and aggregate all labels per unique weight ID.
    self.weight_labels = collate_labels(resolved_weight_ids, shapes, labels)

    # Initialize the bias parameters. Biases are 1D and indexed per output key (one entry per
    # sub-block row), sharing the block structure of the weights' rows. `bias_ids` is the single
    # guard for bias-related state: when biases are disabled, none of `biases`/`bias_shapes`/
    # `bias_labels` are set, so every read of them must be gated on it. `bias_shapes` is parallel
    # to `bias_ids`.
    if self._bias_ids is False:
      self.bias_ids: (
        T.Sequence[T.Sequence[BiasId | T.Literal[0]]] | T.Literal[False]
      ) = False
    else:
      bias_ids: list[BiasId | T.Literal[0] | None] = [
        b for block in self._bias_ids for b in block
      ]
      bias_shapes = [
        (output_size_by_key[o],) for out_group in self.output_keys for o in out_group
      ]
      bias_labels = [
        (output_labels_by_key[o],) for out_group in self.output_keys for o in out_group
      ]
      resolved_bias_ids, self.biases = initialize_shared_weights(
        bias_ids,
        bias_shapes,
        unique_weight_namespace='bias',
      )
      self.bias_ids = []
      self.bias_shapes = []
      cursor = 0
      for out_group in self.output_keys:
        self.bias_ids.append(list(resolved_bias_ids[cursor:cursor + len(out_group)]))
        self.bias_shapes.append(list(bias_shapes[cursor:cursor + len(out_group)]))
        cursor += len(out_group)
      self.bias_labels = collate_labels(resolved_bias_ids, bias_shapes, bias_labels)

    if self.on_initialize:
      self.on_initialize(self)

    self.initialized = True

  def forward(self, inputs: T.Mapping[str, torch.Tensor]) -> T.OrderedDict[str, torch.Tensor]:
    """Returns the outputs, `{output_key: value}`, of each block group: its inputs concatenated in
    order through its assembled weight and bias. The inputs are unbatched 1D tensors: batch with
    `torch.func.vmap`."""
    outputs: T.OrderedDict[str, torch.Tensor] = collections.OrderedDict()
    for i, (in_group, out_group, out_sizes) in enumerate(zip(
      self.input_keys, self.output_keys, self.output_sizes,
    )):
      weight, bias = self.assemble_parameters(i)

      # Concatenate the block's inputs, apply its dense weight (and bias), and split the result back
      # into individual output keys.
      input_tensor = torch.cat([inputs[k] for k in in_group], dim=0)
      output_tensor = input_tensor @ weight.T
      if bias is not None:
        output_tensor = output_tensor + bias

      start = 0
      for k, size in zip(out_group, out_sizes):
        outputs[k] = output_tensor[start:start + size]
        start += size
    return outputs

  def assemble_parameters(self, block_index: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Assembles block `block_index`'s shared sub-block weights (and biases) into a single dense weight
    matrix and bias vector, mirroring `BlockLinear.assemble_parameters`.

    Returns:
      A tuple of (weight_tensor, bias_tensor).
    """
    weight = torch.cat([
      torch.cat([
        get_parameter(self.weights, id).broadcast_to(shape)
        for id, shape in zip(id_row, shape_row)
      ], dim=1)
      for id_row, shape_row in zip(
        self.weight_ids[block_index], self.weight_shapes[block_index],
      )
    ], dim=0)
    bias = torch.cat([
      get_parameter(self.biases, id).broadcast_to(shape)
      for id, shape in zip(self.bias_ids[block_index], self.bias_shapes[block_index])
    ], dim=0) if self.bias_ids else None
    return weight, bias

  def extra_repr(self) -> str:
    # One `BlockLinear`-style grid per block, stacked: output keys label the rows, followed by their
    # biases, then one column per input key of that block.
    if self.initialized:
      weight_ids, bias_ids = self.weight_ids, self.bias_ids
    else:
      weight_ids, bias_ids = self._weight_ids, self._bias_ids
    tables = []
    for i, (in_group, out_group) in enumerate(zip(self.input_keys, self.output_keys)):
      if self.initialized:
        input_headers = [f'{k} ({s})' for k, s in zip(in_group, self.input_sizes[i])]
        output_headers = [f'{k} ({s})' for k, s in zip(out_group, self.output_sizes[i])]
      else:
        input_headers, output_headers = list(in_group), list(out_group)
      table = [[''] + input_headers] + [
        [output_key] + list(row) for output_key, row in zip(output_headers, weight_ids[i])
      ]
      if bias_ids:
        table = [[row[0], bias] + row[1:] for row, bias in zip(table, ['BIAS'] + list(bias_ids[i]))]
      tables.append(table)
    return format_block_table(*tables)

  def __repr__(self) -> str:
    return format_module_repr(self)


# ==================================================================================================
# Initializer factory function

InitializerName = str  # Key of `initializers` in `block_initializer`.
Index = int | str | T.Sequence[int | str]  # Indices or labels along one dimension.
Value = float | T.Sequence[float] | T.Sequence[T.Sequence[float]]  # Values of indexed entries.
Init1D = tuple[Index, Value] | tuple[Index, Value, InitializerName]  # A vector (bias) entry.
Init2D = tuple[Index, Index, Value] | tuple[Index, Index, Value, InitializerName]  # A matrix entry.
# A block-init entry: a per-connection tuple, or a bare `InitializerName` targeting the whole block.
BlockInit2D = Init1D | Init2D | InitializerName
BlockInit1D = Init1D | InitializerName
BlockModule = BlockScaleShift | BlockLinear | SparseBlockLinear


def block_initializer(
  weight_init: T.Mapping[str, T.Sequence[BlockInit2D]],
  bias_init: T.Mapping[str, T.Sequence[BlockInit1D]] | None = None,
  initializers: T.Mapping[
    InitializerName,
    T.Callable[[torch.nn.Parameter | torch.Tensor], None],
  ] | None = None,
) -> T.Callable[[BlockModule], None]:
  """
  Creates a function that initializes the weights and biases of a `BlockModule` (`BlockLinear`,
  `SparseBlockLinear` or `BlockScaleShift`).
  ```
  block_initializer(
    weight_init={
      'front->cross': [
        ('proprio_pos_FL:finger_L', 'body/muscle_flx_FR:shoulder_L_supinate', 0.1, 'freeze'),
      ],
      '*hind': ['zero'],
    },
    bias_init={
      'hind': [
        ('body/muscle_flx_HR:hip_R_extend', -0.1),
      ]
    },
    initializers={
      'freeze': functools.partial(models.constrain, freeze=True),
      'zero': functools.partial(models.constrain, min=0, max=0),
    },
  )
  ```
  Args:
    weight_init: A mapping from weight-identifier globstrings to a list of entries. Keys are matched
      against the module's actual block identifiers `GlobDict`-style (first matching pattern wins),
      and every globstring must match at least one block or a `ValueError` is raised. Each entry is
      either:
        - a connection tuple `(input_label, output_label, value)` or `(..., value,
          initializer_name)`. The labels locate indices in the weight matrix and `value` is
          assigned there; `initializer_name`, if given, applies that initializer to the written
          view. Any label may instead be a sequence of labels/indices, applying the value (and
          initializer) to every combination. `value` is a scalar, or an ndim-nested sequence of
          floats broadcast against the shape of the indexed region. For 1D blocks (e.g.
          `BlockScaleShift`) a connection is simply `(output_label, value[, initializer])`.
        - a bare `initializer_name` (a plain string), which applies that initializer to the entire
          block.
    bias_init: Like `weight_init`, but for bias blocks (always 1D). Raises if given for a module
      with no biases.
    initializers: A mapping from initializer names to functions that take a torch.nn.Parameter and
      apply some initialization to it (e.g., setting it to zero, random values, etc.).
  Returns:
    A function that can be passed to the `on_initialize` argument of a `BlockModule`. This
    function will initialize the weights and biases according to the provided mappings. A
    connection's initializer receives a 1-element view of the parameter (whose `_base` is the
    parameter) for each indexed entry.
  """
  @torch.no_grad()
  def on_initialize(module: BlockModule) -> None:
    _initialize_block_parameters(module.weights, module.weight_labels, weight_init, initializers)
    if module.bias_ids:
      _initialize_block_parameters(
        module.biases, module.bias_labels, bias_init or {}, initializers,
      )
    elif bias_init:
      raise ValueError('bias_init given, but the module has no biases.')
  return on_initialize


def _initialize_block_parameters(
  parameters: torch.nn.ParameterDict,
  labels: T.Mapping[str, T.Sequence[DimLabels]],
  init: T.Mapping[str, T.Sequence[BlockInit2D]],
  initializers: T.Mapping[str, T.Callable[[torch.nn.Parameter | torch.Tensor], None]] | None,
) -> None:
  """
  Initializes a `ParameterDict` of weight or bias blocks, dimensionality-agnostic:
  each block's dimensionality is inferred from `len(labels[id])`.

  Args:
    parameters: A `ParameterDict` of weight or bias blocks, keyed by identifier.
    labels: Per-block, per-dimension index labels, as produced by `collate_labels`.
    init: A mapping from identifier globstrings to a list of entries. Each block identifier is
      matched against the globstrings `GlobDict`-style (first matching pattern wins), and every
      globstring must match at least one block -- an unmatched globstring (e.g. from a typo) raises
      an error rather than silently doing nothing. Each entry is one of:
      - A connection: a tuple of `ndim` indices (each an `int`, `str`, or sequence thereof)
        followed by a value, and optionally an initializer name, where `ndim = len(labels[id])`.
        Connections are conventionally written with indices in reverse dimension order (e.g.
        `(input_index, output_index, value[, initializer])` for a 2D weight block, whose tensor
        dimensions are ordered `(output_dim, input_dim)`); reversing the indices below reconciles
        the two. For a 1D block, reversal is a no-op, so this applies uniformly regardless of
        dimensionality. The value is either a scalar, broadcast to every indexed element, or an
        `ndim`-nested sequence of floats broadcast against the shape of the indexed region (i.e.
        the number of resolved indices per dimension).
      - A bare initializer name (a plain `str`), which applies that initializer to the whole block.
    initializers: A mapping from initializer names to functions that take a `torch.nn.Parameter`
      (or a view of one) and apply some initialization to it.
  """
  block_ids = list(parameters.keys())
  # Resolve each block to the entries of the first init globstring that matches it (or none),
  # erroring on any globstring that matches no block.
  entries_per_block = match_globstrings(
    utc.GlobDict(init), block_ids, disallow_unused_globstrings=True,
    defaults=[() for _ in block_ids],
  )
  for id, entries in zip(block_ids, entries_per_block):
    parameter: torch.nn.Parameter = parameters[id]
    ndim = len(labels[id])
    for entry in entries:
      # A bare initializer name initializes the whole block; anything else is a connection tuple.
      if isinstance(entry, str):
        assert initializers and entry in initializers, f'Initializer {entry!r} not found.'
        initializers[entry](parameter)
        continue
      connection = entry
      if len(connection) == ndim + 1:
        *indices, value = connection
        initializer = None
      elif len(connection) == ndim + 2:
        *indices, value, initializer = connection
      else:
        raise IndexError(
          f'Invalid connection {connection} for block {id!r} with {ndim} dimensions. '
          f'Expected {ndim + 1} or {ndim + 2} elements.'
        )
      indices, value, initializer = (
        T.cast(tuple[Index | None, ...], indices),
        T.cast(Value, value),
        T.cast(str | None, initializer),
      )
      resolved_indices = resolve_indices(indices=indices[::-1], labels=labels[id])
      # Assign `value` (a scalar or an ndim array) to every combination of the resolved indices in
      # one broadcasted write.
      parameter[broadcast_indices(resolved_indices)] = torch.as_tensor(value, dtype=parameter.dtype)
      # Connection-level initializations, applied to a single-element view per combination.
      if initializer:
        assert initializers and initializer in initializers, (
          f'Initializer {initializer!r} not found.'
        )
        for combo in itertools.product(*resolved_indices):
          initializers[initializer](parameter[tuple(slice(i, i + 1) for i in combo)])


# ==================================================================================================
# Label and indexing utilities

def get_flat_box_info(
  space: gymnasium.spaces.Dict,
  keys: T.Sequence[str],
) -> tuple[list[int], list[tuple[str, ...] | None]]:
  """
  Given a gymnasium Dict space and a sequence of keys, verifies that the corresponding values are
  1D Box spaces and extracts their sizes, and if present, dimension labels.

  Args:
    space: A dictionary of gymnasium Box spaces.
    keys: A sequence of keys expected to be present in `space`.

  Returns:
    A tuple of (`sizes`, `labels`), where `sizes` is a list of the sizes of the Box spaces,
    and `labels` is a list of the labels of the Box spaces (or None if no labels are available).
    Both lists are ordered according to `keys`.
  """
  # Validate space keys and types.
  if not all(k in space.spaces for k in keys):
    raise ValueError(
      f'Missing keys: {set(keys) - set(space.spaces)}. '
      f'Space keys: {list(space.spaces.keys())}.'
    )
  if not all(isinstance(space.spaces[k], gymnasium.spaces.Box) for k in keys):
    raise ValueError(f'Got non-Box spaces: {[
      (k, type(space.spaces[k])) for k in keys
      if not isinstance(space.spaces[k], gymnasium.spaces.Box)
    ]}.')
  # Validate box shapes for each key.
  shapes = [space[k].shape for k in keys]
  invalid_shapes = [shape for shape in shapes if shape is None or len(shape) != 1]
  if invalid_shapes:
    raise ValueError(f'Got non-1D shapes: {[
      (k, shape) for k, shape in zip(keys, shapes) if shape is None or len(shape) != 1
    ]}.')
  sizes = [shape[0] for shape in shapes if shape]

  # Extract labels for each key, if available.
  labels: list[tuple[str, ...] | None] = []
  for k in keys:
    box = space.spaces[k]
    if hasattr(box, 'labels'):
      label = getattr(box, 'labels')
      if (
        isinstance(label, T.Sequence) and          # labels=T.Sequence[T.Any]
        len(label) == 1 and                        # labels=[T.Any]
        isinstance(label[0], T.Sequence) and       # labels=[T.Sequence[T.Any]]
        not isinstance(label[0], str) and          # labels!=[str]
        all(isinstance(l, str) for l in label[0])  # labels=[T.Sequence[str]]
      ):
        labels.append(tuple(label[0]))
        continue
    labels.append(None)

  return sizes, labels


def collate_labels(
  weight_ids: T.Sequence[WeightId | T.Literal[0]],
  weight_shapes: T.Sequence[tuple[int, ...]],
  weight_labels: T.Sequence[T.Sequence[T.Sequence[str] | None]],
) -> T.Mapping[WeightId, tuple[DimLabels, ...]]:
  """
  Collate the labels for a set of weights into a single structure.

  Args:
    weight_ids: A sequence of weight IDs.
    weight_shapes: A sequence of shapes for each weight.
    weight_labels: A 2D list of labels that matches the corresponding shape in `weight_shapes`.

  Returns:
    A mapping from unique weight IDs to a `tuple[tuple[frozenset[str], ...], ...]]`, with semantics
    ```
    [
      [set(aliases[index]) for index in range(size)]
      for size, aliases in zip(weight.shape, weight.labels)
    ]
    ```
  """
  # Filter out any disabled weights.
  weight_shapes = [shape for id, shape in zip(weight_ids, weight_shapes) if id != 0]
  weight_labels = [labels for id, labels in zip(weight_ids, weight_labels) if id != 0]
  weight_ids = [id for id in weight_ids if id != 0]

  weight_ids = [parse_weight_transform(w, allow_private=True)[0] for w in weight_ids]
  # Set up the empty indexing structure.
  collated_labels: dict[WeightId, list[list[set[str]]]] = {}
  for weight_id, shape in zip(weight_ids, weight_shapes):
    if weight_id not in collated_labels:
      collated_labels[weight_id] = [[set() for _ in range(dim)] for dim in shape]
      continue
    if len(collated_labels[weight_id]) != len(shape):
      raise ValueError(
        f'Inconsistent number of dimensions for weight {weight_id}: '
        f'{len(collated_labels[weight_id])} vs {len(shape)}.'
      )
    if any(len(dim_labels) != dim for dim_labels, dim in zip(collated_labels[weight_id], shape)):
      raise ValueError(
        f'Inconsistent dimension sizes for weight {weight_id}: '
        f'{[len(dim_labels) for dim_labels in collated_labels[weight_id]]} vs {shape}.'
      )
  # Insert any provided labels into the indexing structure.
  for weight_id, labels in zip(weight_ids, weight_labels):
    if len(labels) != len(collated_labels[weight_id]):
      raise ValueError(
        f'Inconsistent number of dimensions for weight {weight_id}: '
        f'{len(labels)} vs {len(collated_labels[weight_id])}.'
      )
    # `labels` has a length equal to the number of dimensions in the weight.
    for dim_labels, dim_collated in zip(labels, collated_labels[weight_id]):
      # Proceed if no labels were provided for the dimension.
      if dim_labels is None:
        continue
      if len(dim_labels) != len(dim_collated):
        raise ValueError(
          f'Inconsistent dimension sizes for weight {weight_id}: '
          f'{len(dim_labels)} vs {len(dim_collated)}.'
        )
      # Add each label to the corresponding set for each index in the dimension.
      for label, collated in zip(dim_labels, dim_collated):
        collated.add(label)
  # Convert the indexing structure into an immutable version.
  return types.MappingProxyType({
    weight_id: tuple(
      tuple(
        frozenset(dim_labels) for dim_labels in dim_collated
      ) for dim_collated in collated
    ) for weight_id, collated in collated_labels.items()
  })


def resolve_indices(
  indices: T.Sequence[int | str | T.Sequence[int | str] | None],
  labels: T.Sequence[T.Sequence[T.Collection[str]]],
) -> tuple[list[int], ...]:
  """
  Given an `int` or `str` index per dimension, resolve to `int`.
  Multiple indices may be provided for each dimension. A `None` index resolves to every index
  in that dimension.

  Args:
    indices: A sequence specifying an index per dimension.
    labels: A sequence of sequences of label collections, one for each dimension.

  Returns:
    A tuple of lists of resolved indices, one list per dimension.
  """
  resolved_indices: list[list[int]] = []
  for dim_indices, dim_labels in zip(indices, labels):
    if dim_indices is None:
      resolved_indices.append(list(range(len(dim_labels))))
      continue
    if isinstance(dim_indices, (int, str)):
      dim_indices = [dim_indices]
    resolved_dim_indices: list[int] = []
    for index in dim_indices:
      if isinstance(index, int):
        if index < 0 or index >= len(dim_labels):
          raise IndexError(f'Index {index} out of bounds for dimension of size {len(dim_labels)}.')
        resolved_dim_indices.append(index)
        continue
      matched_indices = [i for i, label_set in enumerate(dim_labels) if index in label_set]
      if not matched_indices:
        raise IndexError(f'Label {index} not found in dimension labels: {dim_labels}.')
      if len(matched_indices) > 1:
        raise IndexError(f'Label {index} is ambiguous in dimension labels: {dim_labels}.')
      resolved_dim_indices.append(matched_indices[0])
    resolved_indices.append(resolved_dim_indices)
  return tuple(resolved_indices)


def broadcast_indices(indices: T.Sequence[list[int]]) -> tuple[torch.Tensor, ...]:
  """Reshapes per-dimension index lists (e.g. as returned by `resolve_indices`) for
  `numpy.ix_`-style outer-product indexing: indexing a tensor with the result selects every
  combination of the given per-dimension indices, rather than pairing them elementwise."""
  n = len(indices)
  return tuple(
    torch.tensor(dim_indices).reshape([-1 if d == dim else 1 for d in range(n)])
    for dim, dim_indices in enumerate(indices)
  )


# ==================================================================================================
# Shared weight management utilities

def format_weight_transform(
  weight_id: WeightId | BiasId,
  transform: WeightTransform | None,
) -> WeightId | BiasId:
  """Returns the ID of the `transform` view of weight `weight_id` (e.g. `'-(A)'` for 'negate')."""
  if transform is None:
    return weight_id
  spec = WEIGHT_TRANSFORMS[transform]
  return f'{spec.prefix}({weight_id}){spec.suffix}'


def parse_weight_transform(
  weight_id: WeightId | BiasId,
  allow_private: bool = True,
) -> tuple[WeightId | BiasId, T.Callable[[torch.nn.Parameter | torch.Tensor], torch.Tensor]]:
  """Returns the base weight ID of `weight_id` and the transform that gives its view, e.g. `'-(A)'`
  -> ('A', negation), and 'A' -> ('A', identity).

  Raises:
    ValueError: If the parentheses or transform are invalid, the base ID starts or ends with a
      reserved character, or (unless `allow_private`) it is a generated ID (starts with `$`).
  """
  # Parse transform prefix and suffix.
  prefix_sentinel_parts = weight_id.split('(', 1)
  suffix_sentinel_parts = weight_id.rsplit(')', 1)
  prefix_sentinel_found = len(prefix_sentinel_parts) == 2
  suffix_sentinel_found = len(suffix_sentinel_parts) == 2
  if prefix_sentinel_found != suffix_sentinel_found:
    raise ValueError(f'Mismatched weight transform parentheses in {weight_id}.')

  # No transform on weight ID.
  if not prefix_sentinel_found:
    reserved_characters = set([
      c for c in WEIGHT_ID_RESERVED_CHARACTERS if weight_id.startswith(c) or weight_id.endswith(c)
    ])
    if reserved_characters:
      raise ValueError(
        f'Weight ID {weight_id} starts or ends with reserved characters {reserved_characters}.'
      )
    if not allow_private and weight_id.startswith('$'):
      raise ValueError(f'Weight ID {weight_id} starts with `$`, which only generated IDs may.')
    return weight_id, lambda x: x

  # Identify the transform by matching the prefix and suffix against known transforms.
  prefix, suffix = prefix_sentinel_parts[0], suffix_sentinel_parts[1]
  matching_transforms = [
    (name, spec) for name, spec in WEIGHT_TRANSFORMS.items()
    if prefix == spec.prefix and suffix == spec.suffix
  ]
  if not matching_transforms:
    raise ValueError(f'Invalid weight transform in {weight_id}.')
  assert len(matching_transforms) == 1
  _, spec = matching_transforms[0]

  # Strip the prefix/suffix and parentheses.
  weight_id = weight_id[len(spec.prefix) + 1:-len(spec.suffix) - 1]
  outer_transform = spec.apply

  # Recursively parse nested transforms.
  weight_id, inner_transform = parse_weight_transform(weight_id, allow_private=allow_private)
  transform = lambda x: outer_transform(inner_transform(x))

  return weight_id, transform


def get_parameter(
  parameter_dict: torch.nn.ParameterDict,
  weight_id: WeightId | BiasId | T.Literal[0],
) -> torch.Tensor:
  """Returns the parameter corresponding to a weight ID, with transforms applied."""
  if weight_id == 0:
    return torch.zeros(1)
  weight_id, transform = parse_weight_transform(weight_id, allow_private=True)
  parameter = parameter_dict[weight_id]
  return transform(parameter)


def initialize_shared_weights(
  weight_ids: T.Sequence[WeightId | BiasId | T.Literal[0] | None],
  weight_shapes: T.Sequence[tuple[int, ...]],
  unique_weight_namespace: str,
) -> tuple[T.Sequence[WeightId | BiasId | T.Literal[0]], torch.nn.ParameterDict]:
  """Creates a parameter dict for unique base weight IDs."""
  new_unique_weight_ids = 0
  def unique_weight_id():
    """Returns the next generated ID, `$<namespace><n>`."""
    nonlocal new_unique_weight_ids
    weight_id = f'${unique_weight_namespace}{new_unique_weight_ids}'
    new_unique_weight_ids += 1
    return weight_id

  # Validate that no weight ID is a private ID (starts with `$`).
  for w in weight_ids:
    if isinstance(w, str):
      parse_weight_transform(w, allow_private=False)

  # Replace `None` weight IDs with unique private IDs.
  weight_ids = [
    w if isinstance(w, str) else (
      unique_weight_id() if w is None else w
    ) for w in weight_ids
  ]

  weight_shapes_by_id: dict[WeightId | BiasId, set[tuple[int, ...]]] = {}
  for weight_id, shape in zip(weight_ids, weight_shapes):
    # Disabled (`0`) blocks contribute no parameter, so skip them -- but only after zipping against
    # the full `weight_shapes` so the remaining IDs stay aligned with their shapes.
    if weight_id == 0:
      continue
    base_id = parse_weight_transform(weight_id, allow_private=True)[0]
    weight_shapes_by_id.setdefault(base_id, set()).add(shape)

  inconsistent_weight_shapes = [
    (id, shapes) for id, shapes in weight_shapes_by_id.items() if len(shapes) != 1
  ]
  if inconsistent_weight_shapes:
    raise ValueError(f'Inconsistent weight shapes for identifiers: {inconsistent_weight_shapes}.')

  return weight_ids, torch.nn.ParameterDict({
    weight_id: torch.nn.Parameter(torch.zeros(*next(iter(shapes))))
    for weight_id, shapes in weight_shapes_by_id.items()
  })


# ==================================================================================================
# String utilities

def match_globstrings[V](
  glob_dict: utc.GlobDict[V],
  keys: T.Sequence[str],
  disallow_unused_globstrings: bool,
  defaults: T.Sequence[V] | None = None,
) -> list[V]:
  """
  Matches a sequence of keys against a `GlobDict` of globstrings, returning the corresponding
  values. Raises if any key is unmatched and no default is provided, or if any globstring is unused
  and `disallow_unused_globstrings` is `True`. A globstring shadowed by an earlier, broader one
  (which matches its keys first) counts as unused.
  """
  globstrings = utc.GlobDict(
    {k: k for k in glob_dict.keys()}
  )
  if disallow_unused_globstrings:
    unmatched_globstrings = (set(globstrings.values()) - set([globstrings.get(k) for k in keys]))
    if unmatched_globstrings:
      raise ValueError(
        f'Unused globstrings: {unmatched_globstrings} when matching keys: {keys}'
      )
  if defaults is None:
    if any(k not in glob_dict for k in keys):
      raise ValueError(f'Keys {keys} not found in glob_dict and no defaults provided.')
    defaults = []
  return [glob_dict[k] if k in glob_dict else defaults[i] for i, k in enumerate(keys)]


def format_block_table(*tables: T.Sequence[T.Sequence[T.Any]]) -> str:
  """
  Renders one or more 2D tables (first row of each treated as its header) as aligned, `|`-separated
  strings, for use in the `extra_repr` of a block-structured module (`BlockLinear`,
  `BlockScaleShift`, `SparseBlockLinear`). Multiple tables are stacked with a blank line between
  them, and share column widths so that their columns line up; tables may have different numbers of
  columns. Pair with `format_module_repr` for the `ClassName(...)` wrapping and indent.
  """
  num_columns = max(len(row) for table in tables for row in table)
  max_widths = [
    max(len(str(row[i])) for table in tables for row in table if i < len(row))
    for i in range(num_columns)
  ]
  rendered = []
  for table in tables:
    rows = [
      ' | '.join(str(cell).ljust(width) for cell, width in zip(row, max_widths))
      for row in table
    ]
    table_widths = max_widths[:max(len(row) for row in table)]
    rows.insert(1, '---'.join('-' * width for width in table_widths))
    rendered.append('\n'.join(rows))
  return '\n\n'.join(rendered)


def format_module_repr(module: torch.nn.Module) -> str:
  """
  Renders `module.extra_repr()` wrapped in `ClassName(...)`, in place of `torch.nn.Module`'s default
  `__repr__` -- which would otherwise additionally dump every child submodule (e.g. a `weights` or
  `biases` `ParameterDict`) after `extra_repr`'s own output.
  """
  return f'{module._get_name()}(\n  ' + module.extra_repr().replace('\n', '\n  ') + '\n)'
