import typing as T

import equinox as eqx
import mujoco as mj

import composer as cx

__all__ = ['SKY_COLORS', 'FLOOR_COLORS', 'Floor']

# Skybox gradient colors by name: (upper color, lower color), RGB in [0, 1].
SKY_COLORS = {
  'blue': ((0.3, 0.5, 0.7), (0.0, 0.0, 0.0)),
  'white': ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
  'black': ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
}

# Floor checker colors by name: (light color, dark color), RGB in [0, 1].
FLOOR_COLORS = {
  'blue': ((0.2, 0.3, 0.4), (0.1, 0.2, 0.3)),
  'white': ((0.9, 0.9, 0.9), (0.8, 0.8, 0.8)),
  'gray': ((0.45, 0.44, 0.43), (0.40, 0.39, 0.38)),
  'black': ((0.2, 0.2, 0.2), (0.1, 0.1, 0.1)),
}


class Floor(cx.MjxEntity):
  """A flat floor with a checkered pattern of 1 m squares, a skybox, and a fill light.

  The floor is a plane geom named 'floor', centered at the origin with its surface at z = 0.
  """
  size: tuple[float, float] = eqx.field(static=True)
  reflectance: float = eqx.field(static=True)
  sky_color: T.Literal['blue', 'white', 'black'] = eqx.field(static=True)
  floor_color: T.Literal['blue', 'white', 'gray', 'black'] = eqx.field(static=True)

  def __init__(
    self,
    size: tuple[float, float] = (10., 10.),
    reflectance: float = 0.,
    sky_color: T.Literal['blue', 'white', 'black'] = 'white',
    floor_color: T.Literal['blue', 'white', 'gray', 'black'] = 'gray',
  ):
    """Initializes the floor.

    Args:
      size: Full extent of the floor plane along x and y (m).
      reflectance: Reflectance of the floor material, in [0, 1].
      sky_color: Skybox colors, a key of `SKY_COLORS`.
      floor_color: Checker colors, a key of `FLOOR_COLORS`.
    """
    self.size = size
    self.reflectance = reflectance
    self.sky_color = sky_color
    self.floor_color = floor_color

    super().__init__(mj_spec=self.init_spec())

  def init_spec(self) -> mj.MjSpec:
    """Returns the arena's spec. Subclasses override it to change the spec, e.g. to replace the
    'floor' geom; the fields set in `__init__` before `super().__init__` are available."""
    mj_spec = mj.MjSpec()

    # Visual attributes. The lighting avoids flickering shadows in tracking videos: the body's own
    # tracking spot light (e.g. the rat's) is the only shadow caster, with a cone narrow enough that
    # no shadow-map edge reaches the visible floor; the headlight is flat (no highlights) and the
    # fill light above casts no shadows. Fog in the white of the sky fades the far floor when a
    # render enables it (MuJoCo's 'fog' rendering flag).
    mj_spec.visual.headlight.ambient[:] = (0.75, 0.75, 0.75)
    mj_spec.visual.headlight.diffuse[:] = (0.4, 0.4, 0.4)
    mj_spec.visual.headlight.specular[:] = (0, 0, 0)
    mj_spec.visual.quality.shadowsize = 4096
    mj_spec.visual.map.fogstart = 1.5
    mj_spec.visual.map.fogend = 4.
    mj_spec.visual.rgba.fog[:] = (1, 1, 1, 1)

    # Floor geom.
    mj_spec.add_texture(
      name='floor',
      type=mj.mjtTexture.mjTEXTURE_2D,
      builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
      width=200,
      height=200,
      rgb1=FLOOR_COLORS[self.floor_color][0],
      rgb2=FLOOR_COLORS[self.floor_color][1],
    )
    mj_spec.add_material(
      name='floor',
      texrepeat=(2, 2),  # Makes each square 1x1m since texture has 2x2 squares.
      texuniform=True,
      reflectance=self.reflectance,
      emission=0.7,  # Keeps the floor light without more light on the body.
    ).textures[mj.mjtTextureRole.mjTEXROLE_RGB] = 'floor'
    mj_spec.worldbody.add_geom(
      name='floor',
      type=mj.mjtGeom.mjGEOM_PLANE,
      material='floor',
      size=(self.size[0] / 2, self.size[1] / 2, 0.25),
    )

    # Skybox.
    mj_spec.add_texture(
      name='skybox',
      type=mj.mjtTexture.mjTEXTURE_SKYBOX,
      builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
      rgb1=SKY_COLORS[self.sky_color][0],
      rgb2=SKY_COLORS[self.sky_color][1],
      width=512,
      height=3072,
    )

    # A dim fill light from above, without shadows or highlights.
    mj_spec.worldbody.add_light(
      name='skylight',
      type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
      pos=(0, 0, 2),
      dir=(0, 0, -1),
      diffuse=(0.15, 0.15, 0.15),
      specular=(0, 0, 0),
      castshadow=False,
    )

    return mj_spec
