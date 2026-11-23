import typing as T

import jax
import jax.numpy as jnp


def reward_curve(
  x: jax.Array,
  *,
  optimal: float | jax.Array | tuple[float, float] | tuple[jax.Array, jax.Array],
  margin: float | jax.Array | tuple[float, float] | tuple[jax.Array, jax.Array],
  value_at_optimal: float = 1.,
  value_at_margin: float | None = None,
  curve: T.Literal[
    'linear', 'concave', 'convex', 'cosine', 'gaussian', 'long_tail', 'reciprocal'
  ] = 'linear',
) -> jax.Array:
  """Returns a reward value based on how close `x` is to a target interval.

  Produces the highest reward `value_at_optimal` when `x` is within the inclusive `optimal` range.
  The reward smoothly decreases away from the optimal range, reaching `value_at_margin` at a
  specified `margin` distance on either side of the optimal range. Depending on the chosen `curve`,
  the reward may saturate at `value_at_margin` ('linear', 'convex', 'concave', 'cosine') or
  continue decaying towards zero ('gaussian', 'long_tail').

  Args:
    x: Array of inputs (float).
    optimal: Inclusive lower and upper bounds for the target interval. Defaults to 0. Can be a
      single value (optimal range is a point) or a tuple of (lower, upper) bounds. If bounds are
      specified, it is assumed that elementwise all `lower <= upper`.
    margin: Distance outside the optimal range where the reward decays. Must be non-negative.
      If `0`, the reward is binary: `value_at_optimal` within the optimal range and
      `value_at_margin` outside.
    value_at_optimal: Reward value when `x` is within the optimal range. Defaults to 1.
    value_at_margin: Reward value at the margin distance from the optimal range. If `None`, uses a
      default based on the curve. Ignored if `margin == 0`.
    curve: Shape of the reward decay outside the optimal. Defaults to 'linear'.

  Returns:
    Array of reward values, same shape as `x`.
  """
  # Set default `value_at_margin` based on curve if not provided.
  if value_at_margin is None:
    match curve:
      case 'linear' | 'convex' | 'concave' | 'cosine':
        value_at_margin = 0.
      case 'gaussian' | 'long_tail' | 'reciprocal':
        value_at_margin = 0.2
      case _:
        raise NotImplementedError(f'Unknown curve: {curve}')

  assert 0 <= value_at_margin < value_at_optimal <= 1, (
    f'Expected 0 <= value_at_margin < value_at_optimal <= 1: {value_at_margin=}, '
    f'{value_at_optimal=}'
  )

  # Compute optimal range's lower and upper bounds.
  if isinstance(optimal, tuple):
    assert len(optimal) == 2, f'Expected optimal range to be (lower, upper) bounds: {optimal=}'
    o_lower, o_upper = optimal
  else:
    o_lower, o_upper = optimal, optimal
  o_lower, o_upper = jnp.asarray(o_lower, dtype=float), jnp.asarray(o_upper, dtype=float)

  # Compute margin distance on either side of the optimal range.
  if isinstance(margin, tuple):
    assert len(margin) == 2, f'Expected margin to be (lower, upper) margin: {margin=}'
    m_lower, m_upper = margin
  else:
    m_lower, m_upper = margin, margin
  m_lower, m_upper = jnp.asarray(m_lower, dtype=float), jnp.asarray(m_upper, dtype=float)

  # Compute distance from optimal range, normalized by margin (which may be asymmetric).
  is_optimal = ((o_lower <= x) & (x <= o_upper))
  dist = (
    jnp.nan_to_num((x < o_lower) * (o_lower - x) / m_lower) +
    jnp.nan_to_num((x > o_upper) * (x - o_upper) / m_upper)
  )

  match curve:
    # Finite-support curves that saturate at `value_at_margin` outside the margin.
    case 'linear':
      value = jnp.clip(1 - dist, 0, 1)
      value = value * (value_at_optimal - value_at_margin) + value_at_margin

    case 'concave':
      value = jnp.clip(1 - dist**2, 0, 1)
      value = value * (value_at_optimal - value_at_margin) + value_at_margin

    case 'convex':
      value = jnp.clip((dist < 1) * (1 - dist)**2, 0, 1)
      value = value * (value_at_optimal - value_at_margin) + value_at_margin

    # Infinite-support curves that hit `value_at_margin` at the margin then decay towards zero.
    case 'reciprocal':
      scale = value_at_optimal / value_at_margin - 1
      value = value_at_optimal / (1 + scale * dist)

    case 'long_tail':
      scale = value_at_optimal / value_at_margin - 1
      value = value_at_optimal / (1 + scale * dist**2)

    case _:
      raise NotImplementedError(f'Unknown curve: {curve}')

  value = jnp.nan_to_num(value, nan=value_at_margin)
  return is_optimal * value_at_optimal + ~is_optimal * value


def plot_reward_curves(
  optimal: float | tuple[float, float] = (-1, +1),
  margin: float | tuple[float, float] = 1,
  value_at_margin_infinite: float = 0.2,
  value_at_margin_finite: float = 0.,
  *,
  debug: bool = False,
  xrange: tuple[float, float, int] = (-4, +4, 200),
):
  import matplotlib.pyplot as plt  # Import here to avoid dependency.

  x = jnp.linspace(xrange[0], xrange[1], xrange[2])
  if isinstance(optimal, (int, float)):
    optimal = (optimal, optimal)
  if isinstance(margin, (int, float)):
    margin = (margin, margin)
  vinf = value_at_margin_infinite
  vfin = value_at_margin_finite

  fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(5.5, 3.2))
  for ax in axs:
    ax.axvline(optimal[0], color='0', linestyle='-', linewidth=0.5)
    ax.axvline(optimal[1], color='0', linestyle='-', linewidth=0.5)
    ax.axvline(optimal[0] - margin[0], color='0.3', linestyle='--', linewidth=0.5)
    ax.axvline(optimal[1] + margin[1], color='0.3', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1)

  for curve in ('reciprocal', 'long_tail'):
    y = reward_curve(x, optimal=optimal, margin=margin, value_at_margin=vinf, curve=curve)
    axs[0].plot(x, y, label=curve, clip_on=False)
    if debug: print(f'{curve}:', y)

  for curve in ('linear', 'concave', 'convex'):
    y = reward_curve(x, optimal=optimal, margin=margin, value_at_margin=vfin, curve=curve)
    axs[1].plot(x, y, label=curve, clip_on=False)
    if debug: print(f'{curve}:', y)

  axs[0].set_title(f'Infinite-support:   {optimal=}   {margin=}   value_at_margin={vinf}')
  axs[1].set_title(f'Finite-support:   {optimal=}   {margin=}   value_at_margin={vfin}')
  axs[0].set_ylabel('reward_curve(x)')
  axs[1].set_ylabel('reward_curve(x)')
  axs[1].set_xlabel('x')
  axs[0].legend()
  axs[1].legend()
  axs[0].grid()
  axs[1].grid()
  fig.tight_layout()
  plt.show()
