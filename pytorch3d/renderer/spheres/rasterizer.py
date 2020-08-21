#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from .rasterize_spheres import rasterize_spheres


# Class to store the outputs of point rasterization
class SphereFragments(NamedTuple):
    idx: torch.Tensor
    zbuf: torch.Tensor
    dists: torch.Tensor


# Class to store the point rasterization params with defaults
class SpheresRasterizationSettings:
    __slots__ = [
        "image_size",
        "points_per_pixel",
        "bin_size",
        "max_points_per_bin",
    ]

    def __init__(
        self,
        image_size: int = 256,
        points_per_pixel: int = 8,
        bin_size: Optional[int] = None,
        max_points_per_bin: Optional[int] = None,
    ):
        self.image_size = image_size
        self.points_per_pixel = points_per_pixel
        self.bin_size = bin_size
        self.max_points_per_bin = max_points_per_bin


class SpheresRasterizer(nn.Module):
    def __init__(self, cameras=None, raster_settings=None):
        super().__init__()
        if raster_settings is None:
            raster_settings = SpheresRasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, sphere_clouds, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SpheresRasterizer"
            raise ValueError(msg)

        pts_world = sphere_clouds.points_padded()
        # NOTE: Retaining view space z coordinate for now.
        # TODO: Remove this line when the convention for the z coordinate in
        # the rasterizer is decided. i.e. retain z in view space or transform
        # to a different range.
        pts_view = cameras.get_world_to_view_transform(**kwargs)\
          .transform_points(pts_world)
        pts_screen = cameras.get_projection_transform(**kwargs)\
          .transform_points(pts_view)
        pts_screen[..., 2] = pts_view[..., 2]
        sphere_clouds = sphere_clouds.update_padded(pts_screen)
        return sphere_clouds

    def forward(self, sphere_clouds, **kwargs) -> PointFragments:
        spheres_screen = self.transform(sphere_clouds, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        idx, zbuf, dists2 = rasterize_spheres(
            spheres_screen,
            image_size=raster_settings.image_size,
            points_per_pixel=raster_settings.points_per_pixel,
            bin_size=raster_settings.bin_size,
            max_points_per_bin=raster_settings.max_points_per_bin,
        )
        return SphereFragments(idx=idx, zbuf=zbuf, dists=dists2)
