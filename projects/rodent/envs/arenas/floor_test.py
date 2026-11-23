import mujoco as mj
import numpy as np
import pytest

from envs.arenas import floor


@pytest.mark.parametrize('size', [(10., 10.), (4., 6.)], ids=['default', 'rectangular'])
def test_floor_is_a_plane_named_floor_of_the_given_full_extent(size):
  model = floor.Floor(size=size).mj_spec.compile()

  geom = model.geom('floor')

  assert geom.type[0] == mj.mjtGeom.mjGEOM_PLANE
  np.testing.assert_allclose(geom.size[:2], np.array(size) / 2)
  np.testing.assert_array_equal(geom.pos, [0., 0., 0.])


@pytest.mark.parametrize('sky_color', list(floor.SKY_COLORS))
@pytest.mark.parametrize('floor_color', list(floor.FLOOR_COLORS))
def test_every_named_color_compiles(sky_color, floor_color):
  model = floor.Floor(sky_color=sky_color, floor_color=floor_color).mj_spec.compile()

  assert model.ntex == 2
