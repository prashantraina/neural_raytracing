import numpy as np
import os
import mitsuba
import torch
mitsuba.set_variant('gpu_autodiff_rgb')
#mitsuba.set_variant("scalar_rgb")

from mitsuba.core.xml import load_file, load_string
from mitsuba.core import Thread
from mitsuba.python.autodiff import render_torch, write_bitmap
from mitsuba.python.util import traverse
from pytorch3d.pathtracer.training_utils import ( save_image )

SIZE=512
N=32
THREADS=1
OUTPUT_DIR="."
PER=8
kinds = ["teapot", "buddha", "bunny", "armadillo"]

def elaz_to_xyz(elev, azim, rad):
  elev = np.radians(elev)
  azim = np.radians(azim)
  x = rad * np.cos(elev) * np.sin(azim)
  y = rad * np.cos(elev) * np.cos(azim)
  z = rad * np.sin(elev)
  return x,z,y

def run_mitsuba(min_elev, max_elev, min_azim, max_azim, radius):
  for k in kinds:
    for i, elev in enumerate(np.linspace(min_elev, max_elev, PER)):
      for j, azim in enumerate(np.linspace(min_azim, max_azim, PER)):
        ox, oy, oz = elaz_to_xyz(elev, azim, radius)
        lx, ly, lz = elaz_to_xyz(elev, azim, 1.05 * radius)
        scene = get_scene(ox, oy, oz, lx, ly, lz, k)
        out = render_torch(scene, spp=32)
        for _ in range(N-1):
          out += render_torch(scene, spp=32)
        out /= N
        out = torch.cat([
          out[..., :3],
          out[..., 3].unsqueeze(-1) > 0,
        ], dim=-1)
        save_image(f"{k}_{i:03}_{j:03}.png", out)

def get_scene(ox, oy, oz, lx, ly, lz, k):
  lx = 275 + lx
  ly = 275 + ly
  lz = 280 + lz
  assert(k in kinds)
  return load_string(f"""
<scene version="2.2.1">
    <integrator type="aov">
      <string name="aovs" value="dd.y:depth"/>
      <integrator type="direct"/>
    </integrator>

    <sensor type="perspective">
        <string name="fov_axis" value="smaller"/>
        <float name="near_clip" value="0.01"/>
        <float name="far_clip" value="10000"/>
        <float name="fov" value="60"/>
        <transform name="to_world">
            <lookat origin="{275 + ox}, {275 + oy}, {280 + oz}"
                    target="275, 275, 280"
                    up    ="  0,   1,    0.00001"/>
        </transform>
        <sampler type="independent">  <!-- ldsampler -->
            <integer name="sample_count" value="128"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="256"/>
            <integer name="height" value="256"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    <emitter type="point">
      <spectrum name="intensity" value="5000000"/>
      <point name="position" x="{lx}" y="{ly}" z="{lz}"/>
    </emitter>

    <bsdf type="twosided" id="red">
      <bsdf type="roughplastic">
        <string name="distribution" value="beckmann"/>
        <float name="int_ior" value="1.61"/>
        <rgb name="diffuse_reflectance" value="0.5, 0, 0"/>
      </bsdf>
    </bsdf>
    <bsdf type="twosided" id="green">
      <bsdf type="plastic">
        <rgb name="diffuse_reflectance" value="0.1, 0.8, 0.36"/>
        <float name="int_ior" value="1.1"/>
      </bsdf>
    </bsdf>
    <bsdf type="twosided" id="white">
      <bsdf type="diffuse">
        <rgb name="reflectance" value="0.2, 0.25, 0.7"/>
      </bsdf>
    </bsdf>
    <bsdf type="twosided" id="box">
      <bsdf type="roughconductor">
        <string name="material" value="Au"/>
        <string name="distribution" value="ggx"/>
        <float name="alpha_u" value="0.05"/>
        <float name="alpha_v" value="0.3"/>
      </bsdf>
    </bsdf>
    {"<!--" if k != "teapot" else ""}
    <shape type="obj">
      <string name="filename" value="teapot.obj"/>
      <transform name="to_world">
        <scale value="150"/>
        <translate x="255" y="0" z="250"/>
      </transform>
      <bsdf type="diffuse">
        <rgb name="reflectance" value="0.7, 0.25, 0.1"/>
      </bsdf>
    </shape>
    {"-->" if k != "teapot" else ""}

    {"<!--" if k != "armadillo" else ""}
    <shape type="obj">
      <string name="filename" value="armadillo.obj"/>
      <transform name="to_world">
        <scale value="300"/>
        <translate x="250" y="200" z="250"/>
      </transform>
      <ref id="box"/>
    </shape>
    {"-->" if k != "armadillo" else ""}

    {"<!--" if k != "bunny" else ""}
    <shape type="obj">
      <string name="filename" value="bunny.obj"/>
      <transform name="to_world">
        <scale value="4000"/>
        <translate x="250" y="-200" z="250"/>
      </transform>
      <ref id="green"/>
    </shape>
    {"-->" if k != "bunny" else ""}

    {"<!--" if k != "buddha" else ""}
    <shape type="obj">
      <string name="filename" value="happy.obj"/>
      <transform name="to_world">
        <scale value="6000"/>
        <translate x="250" y="-700" z="250"/>
      </transform>
      <bsdf type="roughplastic">
        <string name="distribution" value="beckmann"/>
        <float name="int_ior" value="1.61"/>
        <rgb name="diffuse_reflectance" value="0.8,0.1,0.1"/>
      </bsdf>
    </shape>
    {"-->" if k != "buddha" else ""}
</scene>
  """)

if __name__ == "__main__":
  scene = run_mitsuba(0, 45, -90, 90, 800)
