import jax.numpy as jnp
import numpy as np
import pytest

from composer.utils import rewards

# ==================================================================================================
# Reward curves

FINITE_CURVES = ['linear', 'concave', 'convex']
INFINITE_CURVES = ['reciprocal', 'long_tail']

# An optimal range [1, 2] with asymmetric margins: 0.5 below, 2 above.
OPTIMAL = (1.0, 2.0)
MARGIN = (0.5, 2.0)


@pytest.mark.parametrize('curve', FINITE_CURVES + INFINITE_CURVES)
def test_reward_curve_is_value_at_optimal_in_range_and_value_at_margin_at_margins(curve):
  x = jnp.array([1.0, 1.5, 2.0, 0.5, 4.0])  # Range ends and middle, then each margin.

  value = rewards.reward_curve(
    x,
    optimal=OPTIMAL,
    margin=MARGIN,
    value_at_optimal=0.9,
    value_at_margin=0.3,
    curve=curve,
  )

  np.testing.assert_allclose(value, [0.9, 0.9, 0.9, 0.3, 0.3], atol=1e-6)


@pytest.mark.parametrize('curve', FINITE_CURVES + INFINITE_CURVES)
@pytest.mark.parametrize(
  'x',
  [jnp.linspace(1.0, -5.0, 61), jnp.linspace(2.0, 10.0, 81)],
  ids=['below', 'above'],
)
def test_reward_curve_is_bounded_and_decreases_away_from_optimal(curve, x):
  value = np.asarray(rewards.reward_curve(x, optimal=OPTIMAL, margin=MARGIN, curve=curve))

  assert np.all((0 <= value) & (value <= 1))
  assert np.all(np.diff(value) <= 1e-7)  # `x` moves away from the range.


@pytest.mark.parametrize('curve', FINITE_CURVES)
def test_finite_reward_curves_stay_at_value_at_margin_beyond_the_margin(curve):
  x = jnp.array([-1.0, 0.0, 5.0, 10.0])

  value = rewards.reward_curve(x, optimal=OPTIMAL, margin=MARGIN, value_at_margin=0.1, curve=curve)

  np.testing.assert_allclose(value, 0.1, atol=1e-6)


@pytest.mark.parametrize('curve', INFINITE_CURVES)
def test_infinite_reward_curves_decay_below_value_at_margin_beyond_the_margin(curve):
  x = jnp.array([0.0, -1.0, 5.0, 10.0])

  value = np.asarray(
    rewards.reward_curve(x, optimal=OPTIMAL, margin=MARGIN, value_at_margin=0.2, curve=curve)
  )

  assert np.all((0 < value) & (value < 0.2))
  assert value[1] < value[0] and value[3] < value[2]


@pytest.mark.parametrize(
  'curve, value_at_margin',
  [('linear', 0.0), ('reciprocal', 0.2), ('long_tail', 0.2)],
)
def test_reward_curve_defaults_value_at_margin_by_curve(curve, value_at_margin):
  value = rewards.reward_curve(jnp.array(0.5), optimal=OPTIMAL, margin=MARGIN, curve=curve)

  np.testing.assert_allclose(value, value_at_margin, atol=1e-6)


@pytest.mark.parametrize('curve', FINITE_CURVES)
def test_finite_reward_curves_with_zero_margin_are_binary(curve):
  x = jnp.array([0.0, 1.0, 1.5, 2.0, 3.0])

  value = rewards.reward_curve(x, optimal=OPTIMAL, margin=0.0, value_at_margin=0.1, curve=curve)

  np.testing.assert_allclose(value, [0.1, 1.0, 1.0, 1.0, 0.1], atol=1e-6)


def test_reward_curve_targets_a_point_when_optimal_is_a_scalar():
  value = rewards.reward_curve(jnp.array([0.0, 0.5, 1.0]), optimal=0.0, margin=1.0)

  np.testing.assert_allclose(value, [1.0, 0.5, 0.0], atol=1e-6)


@pytest.mark.parametrize(
  'value_at_optimal, value_at_margin',
  [(0.5, 0.5), (0.5, 0.8), (1.5, 0.2)],
  ids=['equal', 'margin_above_optimal', 'optimal_above_one'],
)
def test_reward_curve_rejects_invalid_values(value_at_optimal, value_at_margin):
  with pytest.raises(AssertionError):
    rewards.reward_curve(
      jnp.array(0.0),
      optimal=0.0,
      margin=1.0,
      value_at_optimal=value_at_optimal,
      value_at_margin=value_at_margin,
    )
