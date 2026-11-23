import mujoco as mj
import numpy as np
import pytest

from envs.arenas import bumps


def compile_terrain(**kwargs) -> mj.MjModel:
  """Returns the compiled model of a seeded `Bumps`."""
  return bumps.Bumps(**(dict(seed=0) | kwargs)).mj_spec.compile()


@pytest.mark.parametrize(
  'kwargs',
  [
    dict(),
    dict(size=(6, 6), bump_height=0.02, bump_scale=0.05, resolution=50),
    dict(size=(6, 6), bump_height=0.04, bump_scale=0.05, resolution=50),
  ],
  ids=['default', 'height_0.02', 'height_0.04'],
)
def test_terrain_spans_the_half_extent_and_bump_height_below_zero(kwargs):
  arena = bumps.Bumps(**(dict(seed=0) | kwargs))
  model = arena.mj_spec.compile()

  geom = model.geom('floor')
  heights = geom.pos[2] + model.hfield_size[0, 2] * model.hfield_data

  assert geom.type[0] == mj.mjtGeom.mjGEOM_HFIELD
  assert model.hfield_nrow[0] == model.hfield_ncol[0] == int(arena.resolution * arena.size[0])
  np.testing.assert_array_equal(model.hfield_size[0, :2], arena.size)
  np.testing.assert_allclose([heights.min(), heights.max()], [-arena.bump_height, 0.], atol=1e-7)


def test_zero_bump_height_gives_bumps_of_the_least_height():
  model = compile_terrain(size=(1, 1), bump_height=0., bump_scale=0.05, resolution=50)

  heights = model.geom('floor').pos[2] + model.hfield_size[0, 2] * model.hfield_data

  np.testing.assert_allclose([heights.min(), heights.max()], [0., 1e-5], atol=1e-9)


def test_seed_determines_the_terrain():
  kwargs = dict(size=(1, 1), bump_height=0.02, bump_scale=0.05, resolution=50)

  first, again, other = (compile_terrain(seed=seed, **kwargs) for seed in (0, 0, 1))

  np.testing.assert_array_equal(first.hfield_data, again.hfield_data)
  assert not np.array_equal(first.hfield_data, other.hfield_data)


def test_no_seed_draws_from_the_global_random_state():
  kwargs = dict(size=(1, 1), bump_height=0.02, bump_scale=0.05, resolution=50, seed=None)

  models = []
  for _ in range(2):
    np.random.seed(0)
    models.append(bumps.Bumps(**kwargs).mj_spec.compile())

  np.testing.assert_array_equal(models[0].hfield_data, models[1].hfield_data)
