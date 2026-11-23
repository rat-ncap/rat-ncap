import typing as T

import jax
import jax.numpy as jnp
import numpy as np

# ==================================================================================================
# Rescaling


def rescale_linear2(
  x: jax.Array,
  x1: jax.Array | float,
  x2: jax.Array | float,
  y1: jax.Array | float,
  y2: jax.Array | float,
  *,
  clip: bool = False,
) -> jax.Array:
  """Linearly rescales `x` from [x1, x2] to [y1, y2], extrapolating outside the interval.

  Args:
    x: Values to rescale.
    x1: Input value that maps to `y1`.
    x2: Input value that maps to `y2`.
    y1: Output value at `x1`.
    y2: Output value at `x2`.
    clip: Whether to clip the outputs to the range between `y1` and `y2`.

  Returns:
    The rescaled values. A non-finite result, such as every output when `x1 == x2`, becomes the
    midpoint `(y1 + y2) / 2`.
  """
  y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
  y12 = (y1 + y2) / 2
  y = jnp.nan_to_num(y, nan=y12, posinf=y12, neginf=y12)  # Handle x1 == x2.
  if clip:
    y = jnp.clip(y, jnp.minimum(y1, y2), jnp.maximum(y1, y2))
  return y


def rescale_linear3(
  x: jax.Array,
  x1: jax.Array | float,
  x2: jax.Array | float,
  x3: jax.Array | float,
  y1: jax.Array | float,
  y2: jax.Array | float,
  y3: jax.Array | float,
  *,
  clip: bool = False,
) -> jax.Array:
  """Piecewise-linearly rescales `x` from [x1, x2, x3] to [y1, y2, y3].

  Values below `x2` use the segment [x1, x2] -> [y1, y2] and the rest use [x2, x3] -> [y2, y3],
  each extrapolating beyond its end.

  Args:
    x: Values to rescale.
    x1: Input value that maps to `y1`.
    x2: Input value that maps to `y2`, the breakpoint.
    x3: Input value that maps to `y3`.
    y1: Output value at `x1`.
    y2: Output value at `x2`.
    y3: Output value at `x3`.
    clip: Whether to clip the outputs to the range spanned by `y1`, `y2` and `y3`.

  Returns:
    The rescaled values. A non-finite result, such as for a value on a zero-width segment
    (`x1 == x2` or `x2 == x3`), becomes `y2`.
  """
  left = (x < x2) * (y1 + (y2 - y1) * (x - x1) / (x2 - x1))
  right = (x >= x2) * (y2 + (y3 - y2) * (x - x2) / (x3 - x2))
  y = jnp.nan_to_num(left + right, nan=y2, posinf=y2, neginf=y2)  # Handle x1 == x2 or x2 == x3.
  if clip:
    y = jnp.clip(y, jnp.minimum(jnp.minimum(y1, y2), y3), jnp.maximum(jnp.maximum(y1, y2), y3))
  return y


def interpolate_linear2(
  t: jax.Array,
  y1: jax.Array | float,
  y2: jax.Array | float,
  *,
  clip: bool = False,
) -> jax.Array:
  """Linearly maps `t` from [0, 1] to [y1, y2] (see `rescale_linear2`)."""
  return rescale_linear2(t, 0, 1, y1, y2, clip=clip)


def interpolate_linear3(
  t: jax.Array,
  y1: jax.Array | float,
  y2: jax.Array | float,
  y3: jax.Array | float,
  *,
  clip: bool = False,
) -> jax.Array:
  """Piecewise-linearly maps `t` from [-1, 0, 1] to [y1, y2, y3] (see `rescale_linear3`)."""
  return rescale_linear3(t, -1, 0, 1, y1, y2, y3, clip=clip)


# ==================================================================================================
# Tuples

ArrayLike = float | int | np.ndarray | T.Sequence[float | int]
"""A scalar or a 1-D sequence or array of numbers."""


def float_tuple(x: ArrayLike) -> tuple[float, ...]:
  """Converts a scalar or 1-D sequence to a tuple of floats (a scalar becomes a 1-tuple)."""
  return tuple(np.atleast_1d(x).astype(float).tolist())


def int_tuple(x: ArrayLike) -> tuple[int, ...]:
  """Converts a scalar or 1-D sequence to a tuple of ints (a scalar becomes a 1-tuple).

  Floats are truncated toward zero.
  """
  return tuple(np.atleast_1d(x).astype(int).tolist())


# ==================================================================================================
# Rotations


def quat_to_rpy(quat: jax.Array) -> jax.Array:
  """Converts a quaternion to roll, pitch and yaw angles.

  The angles are intrinsic ZYX (yaw, then pitch, then roll), so drift in yaw (Z) does not affect
  roll (X) and pitch (Y).

  Args:
    quat: Unit quaternion `[w, x, y, z]` (MuJoCo's convention). Shape `[4]`.

  Returns:
    `[roll, pitch, yaw]` in radians, with roll and yaw in [-pi, pi] and pitch in [-pi/2, pi/2].
    Shape `[3]`.
  """
  w, x, y, z = quat
  roll = jnp.arctan2(2. * (w * x + y * z), 1. - 2. * (x**2 + y**2))
  pitch = jnp.arcsin(2. * (w * y - z * x))
  yaw = jnp.arctan2(2. * (w * z + x * y), 1. - 2. * (y**2 + z**2))
  return jnp.array([roll, pitch, yaw])


def rpy_to_quat(rpy: jax.Array) -> jax.Array:
  """Converts roll, pitch and yaw angles to a quaternion, the inverse of `quat_to_rpy`.

  Args:
    rpy: `[roll, pitch, yaw]` in radians, intrinsic ZYX. Shape `[3]`.

  Returns:
    Unit quaternion `[w, x, y, z]`. Shape `[4]`.
  """
  roll, pitch, yaw = rpy
  cy = jnp.cos(yaw * 0.5)
  sy = jnp.sin(yaw * 0.5)
  cp = jnp.cos(pitch * 0.5)
  sp = jnp.sin(pitch * 0.5)
  cr = jnp.cos(roll * 0.5)
  sr = jnp.sin(roll * 0.5)
  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy
  return jnp.array([w, x, y, z])


def quat_to_xmat(quat: jax.Array) -> jax.Array:
  """Converts a quaternion to a rotation matrix, as MuJoCo's `mju_quat2Mat`.

  Args:
    quat: Unit quaternion `[w, x, y, z]`. A non-unit quaternion gives a scaled, non-orthogonal
      matrix. Shape `[4]`.

  Returns:
    Rotation matrix that maps body-frame vectors to the world frame (MuJoCo's `xmat`, unflattened).
    Shape `[3, 3]`.
  """
  q = jnp.outer(quat, quat)
  return jnp.array([
    [
      q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
      2 * (q[1, 2] - q[0, 3]),
      2 * (q[1, 3] + q[0, 2]),
    ],
    [
      2 * (q[1, 2] + q[0, 3]),
      q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
      2 * (q[2, 3] - q[0, 1]),
    ],
    [
      2 * (q[1, 3] - q[0, 2]),
      2 * (q[2, 3] + q[0, 1]),
      q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
    ],
  ])
