# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from .blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from .cameras import OpenGLOrthographicCameras  # deprecated
from .cameras import OpenGLPerspectiveCameras  # deprecated
from .cameras import SfMOrthographicCameras  # deprecated
from .cameras import SfMPerspectiveCameras  # deprecated
from .cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    camera_position_from_spherical_angles,
    get_world_to_view_transform,
    look_at_rotation,
    look_at_view_transform,
)
from .lighting import DirectionalLights, PointLights, diffuse, specular
from .materials import Materials
from .mesh import (
    NeuralShader,
    DebugShader,

    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftGouraudShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    Textures,
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
    gouraud_shading,
    phong_shading,
    rasterize_meshes,
)
from .points import (
    AlphaCompositor,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PulsarPointsRenderer,
    rasterize_points,
)
from .utils import TensorProperties, convert_to_tensors_and_broadcast

__all__ = [k for k in globals().keys() if not k.startswith("_")]
