# Rat

`rat.xml` is the MuJoCo model (MJCF) of the rat body used by `rat.py`. Its skin (`rat.skn`) is
drawn over the simulated bodies, with one of three coat textures (`rat_blackhooded.png`, the
default, `rat_cinnamon.png` and `rat_gray.png`).

## Origin and modifications

The model is derived from the rodent model of
[google-deepmind/dm_control](https://github.com/google-deepmind/dm_control), released under the
[Apache License 2.0](LICENSE). We modified it for this work, including:

- recalculated inertial properties;
- tail geom simplification;
- an adjusted default pose, joint ranges and actuators for the joint-level control interfaces;
- sensors for proprioception, touch, and the feet's global positions and velocities;
- collision geoms for contact and fall detection;
- a new skin and textures (below).

## Skin and textures

The skin (`rat.skn`) and textures (`rat_*.png`) were created by the authors and are released under
this repository's MIT License.
