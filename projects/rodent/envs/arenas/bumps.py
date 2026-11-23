import typing as T

import equinox as eqx
import mujoco as mj
import numpy as np
import scipy.ndimage

from .floor import Floor

__all__ = ['Bumps']


class Bumps(Floor):
  """A bumpy terrain, with the skybox and lights of `Floor`.

  The terrain is a heightfield geom named 'floor', centered at the origin, that replaces the flat
  floor. Its heights are random knots `bump_scale` apart, smoothly interpolated and scaled to span
  [-`bump_height`, 0] (m): the highest point is at z = 0. A `bump_height` of 0 gives bumps 1e-5 m
  high, spanning [0, 1e-5], as a MuJoCo heightfield needs a positive height.
  """
  bump_height: float = eqx.field(static=True)
  bump_scale: float = eqx.field(static=True)
  resolution: float = eqx.field(static=True)
  seed: int | None = eqx.field(static=True)

  def __init__(
    self,
    size: tuple[float, float] = (10, 10),
    bump_height: float = 0.005,
    bump_scale: float = 0.03,
    resolution: float = 200,
    seed: int | None = None,
  ):
    """Initializes the terrain.

    Args:
      size: Half-extent of the terrain along x and y (m): it spans [-size, size]. Must be square.
      bump_height: Height difference between the lowest and the highest point (m), in [0, 1].
      bump_scale: Distance between the random height knots (m).
      resolution: Heightfield samples along each axis per meter of `size`.
      seed: Seed of the random heights, or None to draw them from NumPy's global random state.
    """
    assert size[0] == size[1]  # Needs to be square.
    assert 0 <= bump_height <= 1
    assert 0 < bump_scale
    self.bump_height = bump_height
    self.bump_scale = bump_scale
    self.resolution = resolution
    self.seed = seed

    super().__init__(size=size)

  @T.override
  def init_spec(self) -> mj.MjSpec:
    """Returns the spec of `Floor` with the flat floor replaced by the terrain."""
    mj_spec = super().init_spec()
    mj_spec.delete(mj_spec.geom('floor'))

    nrow = int(self.resolution * self.size[1])
    ncol = int(self.resolution * self.size[0])
    assert nrow == ncol
    res = nrow
    terrain_size = 2 * self.size[0]
    bump_res = int(terrain_size / self.bump_scale)
    rng = np.random if self.seed is None else np.random.default_rng(self.seed)
    bumps = rng.uniform(0, 1, (bump_res, bump_res))
    terrain: np.ndarray = scipy.ndimage.zoom(bumps, res / float(bump_res))

    # MuJoCo rescales the heights to [0, 1], times the elevation `bump_height`.
    mj_spec.add_hfield(
      name='terrain',
      nrow=nrow,
      ncol=ncol,
      size=(*self.size, max(self.bump_height, 1e-5), 0.1),
      userdata=terrain.flatten().tolist(),
    )
    mj_spec.worldbody.add_geom(
      name='floor',
      type=mj.mjtGeom.mjGEOM_HFIELD,
      rgba=(0.2, 0.3, 0.4, 1),
      pos=(0, 0, -self.bump_height),
      hfieldname='terrain',
    )

    return mj_spec
