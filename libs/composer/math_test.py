import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from composer import math

# ==================================================================================================
# Rescaling


def test_rescale_linear2_maps_endpoints_and_extrapolates():
  x = jnp.array([0.0, 1.0, 2.0, 3.0])

  y = math.rescale_linear2(x, 1.0, 2.0, 10.0, 20.0)

  np.testing.assert_allclose(y, [0.0, 10.0, 20.0, 30.0])


def test_rescale_linear2_clip_keeps_outputs_between_y1_and_y2():
  x = jnp.array([0.0, 1.5, 3.0])

  y = math.rescale_linear2(x, 1.0, 2.0, 20.0, 10.0, clip=True)

  np.testing.assert_allclose(y, [20.0, 15.0, 10.0])


def test_rescale_linear2_maps_zero_width_interval_to_midpoint():
  x = jnp.array([0.0, 1.0, 2.0])

  y = math.rescale_linear2(x, 1.0, 1.0, 10.0, 20.0)

  np.testing.assert_allclose(y, [15.0, 15.0, 15.0])


def test_rescale_linear3_maps_breakpoints_and_interpolates_each_segment():
  x = jnp.array([-1.0, -0.5, 0.0, 1.0, 2.0])

  y = math.rescale_linear3(x, -1.0, 0.0, 2.0, -2.0, 0.0, 4.0)

  np.testing.assert_allclose(y, [-2.0, -1.0, 0.0, 2.0, 4.0])


def test_rescale_linear3_clip_keeps_outputs_within_y_range():
  x = jnp.array([-5.0, 5.0])

  y = math.rescale_linear3(x, -1.0, 0.0, 1.0, -2.0, 0.0, 4.0, clip=True)

  np.testing.assert_allclose(y, [-2.0, 4.0])


@pytest.mark.parametrize(
  'x1, x2, x3, expected',
  [
    (0.0, 0.0, 1.0, [0.0, 0.0, 0.5, 1.0]),
    (-1.0, 0.0, 0.0, [-1.0, 0.0, 0.0, 0.0]),
  ],
  ids=['zero_width_left', 'zero_width_right'],
)
def test_rescale_linear3_keeps_valid_segment_when_other_has_zero_width(x1, x2, x3, expected):
  # The zero-width segment maps to y2; the other segment still interpolates.
  x = jnp.array([-1.0, 0.0, 0.5, 1.0])

  y = math.rescale_linear3(x, x1, x2, x3, -1.0, 0.0, 1.0)

  np.testing.assert_allclose(y, expected)


def test_interpolate_linear3_maps_minus_one_zero_one_to_y_values():
  t = jnp.array([-1.0, 0.0, 1.0])

  y = math.interpolate_linear3(t, 5.0, 6.0, 8.0)

  np.testing.assert_allclose(y, [5.0, 6.0, 8.0])


# ==================================================================================================
# Tuples


@pytest.mark.parametrize(
  'x, expected',
  [(2, (2.0,)), ([1, 2.5], (1.0, 2.5)), (np.array([3, 4]), (3.0, 4.0))],
  ids=['scalar', 'list', 'array'],
)
def test_float_tuple_converts_scalars_and_sequences(x, expected):
  assert math.float_tuple(x) == expected


def test_int_tuple_truncates_floats_toward_zero():
  assert math.int_tuple([1.7, -1.7]) == (1, -1)


# ==================================================================================================
# Rotations


def random_rpy(seed: int) -> np.ndarray:
  """Random roll, yaw in (-pi, pi) and pitch away from the +-pi/2 singularity."""
  rng = np.random.default_rng(seed)
  return np.array([
    rng.uniform(-np.pi, np.pi),
    rng.uniform(-1.4, 1.4),
    rng.uniform(-np.pi, np.pi),
  ])


def mujoco_quat_zyx(rpy: np.ndarray) -> np.ndarray:
  """MuJoCo's quaternion for an intrinsic ZYX rotation: yaw about z, then pitch about y, then roll
  about x."""
  quats = []
  for axis, angle in zip(np.eye(3)[::-1], rpy[::-1]):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    quats.append(q)
  q_zy, q_zyx = np.zeros(4), np.zeros(4)
  mujoco.mju_mulQuat(q_zy, quats[0], quats[1])
  mujoco.mju_mulQuat(q_zyx, q_zy, quats[2])
  return q_zyx


@pytest.mark.parametrize('seed', range(5))
def test_rpy_to_quat_matches_mujoco_intrinsic_zyx_rotation(seed):
  rpy = random_rpy(seed)

  quat = math.rpy_to_quat(jnp.array(rpy))

  np.testing.assert_allclose(quat, mujoco_quat_zyx(rpy), atol=1e-6)


@pytest.mark.parametrize('seed', range(5))
def test_quat_to_rpy_inverts_rpy_to_quat(seed):
  rpy = random_rpy(seed)

  restored = math.quat_to_rpy(math.rpy_to_quat(jnp.array(rpy)))

  np.testing.assert_allclose(restored, rpy, atol=1e-5)


@pytest.mark.parametrize('seed', range(5))
def test_quat_to_xmat_matches_mujoco_quat2mat(seed):
  quat = mujoco_quat_zyx(random_rpy(seed))
  expected = np.zeros(9)
  mujoco.mju_quat2Mat(expected, quat)

  xmat = math.quat_to_xmat(jnp.array(quat))

  np.testing.assert_allclose(xmat, expected.reshape(3, 3), atol=1e-6)
