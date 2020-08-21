# Pathtracer

Extension to include raytracing. Does not directly rely on other PyTorch3D packages except for
the `camera` module. The entrypoint to this module is [`main.py`](main.py), which defines
methods for sampling from a scene, or rendering a full image.

Inside of [`scene.py`](scene.py), we have a bunch of functions utilized inside of integrators
for scene rendering, including light sampling w or w/o intersection, or with a learned occlusion
parameter. It also includes a vectorized mesh intersection algorithm, which was used while
developing other parts of the framework.

[`neural_blocks.py`](neural_blocks.py) contains the definition of models used in our approach.
This includes an MLP with Fourier features and skip connections. And a two-stage MLP akin to
NeRF.

[`utils.py`](utils.py) is where a bunch of utility and conversion functions are, including loss
functions, etc. [`training_utils.py`](training_utils.py) contains functions for training loops
over a variety of datasets.

[`warps.py`](warps.py) contains functions for mapping distributions, many of which are taken
directly from [Mitsuba](https://mitsuba2.readthedocs.io/en/latest/).

[`interaction.py`](interaction.py) is where surface intersections are defined which are used
internally for passing information from the surface to the integrator about details of the
interaction, including normal, point of intersection, depth, and also holds information which
can be used for back-propagation.

## Submodules

Each of the submodules, `bsdf`, `cameras`, `shapes`, `lights`, and `integrators`, contains a
single file which contains most of the functionality of the module. `samplers` is currently
unused, as we mainly rely on uniform distributions.

