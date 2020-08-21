import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
import pytorch3d.pathtracer as pt
from pytorch3d.pathtracer.bsdf import ( ComposeSpatialVarying, NeuralBSDF )
from pytorch3d.pathtracer.integrators import (
  BasisBRDF, Debug, Depth, Direct, NeRFIntegrator, Illumination,
)
from pytorch3d.pathtracer.utils import ( sphere_examples, heightmap, count_parameters )
from pytorch3d.pathtracer.shapes.sdfs import SDF
from pytorch3d.renderer import (
  look_at_view_transform, OpenGLPerspectiveCameras, PointLights, HardPhongShader,
)
import math
from tqdm import trange

SIZE=256
N_VIEWS=9
DIST=0.9

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def save_image(name, img):
  plt.imsave(name, img.detach().cpu().clamp(0, 1).numpy())

perspectives = []
a, b = torch.meshgrid(torch.linspace(0.1, 360, N_VIEWS), torch.linspace(0.1, 360, N_VIEWS))
for i in range(N_VIEWS):
  thru = range(N_VIEWS)
  # just looks nicer to go back and forth
  if i % 2 == 0: thru = reversed(thru)
  for j in thru:
    perspectives.append(look_at_view_transform(dist=DIST, elev=a[i, j], azim=b[i, j]))


integrator=Direct()

dataset="materials"
print(dataset)

shape = torch.jit.load(f"models/{dataset}_sdf_f.pt", device)
shape = SDF(sdf=shape)
print("Number of shape parameters:", count_parameters(shape.parameters()))

shape.max_steps=128

bsdf = torch.load(f"models/{dataset}_bsdf_f.pt")
print("Number of bsdf parameters:", count_parameters(bsdf.parameters()))
#for b in bsdf.bsdfs: setattr(b, 'act', torch.sigmoid)

lights = torch.load(f"models/{dataset}_light_f.pt")
print("Number of light parameters:", count_parameters(lights.parameters()))

def naive_squarest_divisor(v):
  sqrt = math.sqrt(v)
  curr = 1
  for i in range(2, v+1):
    if (v % i == 0) and (abs(sqrt-i) < abs(sqrt - curr)):
      curr = i
  return curr, v//curr

def unroll(i, w):
  return i//w, i % w

if isinstance(bsdf, ComposeSpatialVarying):
  N = len(bsdf.bsdfs)
  r, c = naive_squarest_divisor(N)
  print("Using axes", r, c)
  assert(r * c == N)
  with torch.no_grad():
    f, axes = plt.subplots(r, c)
    for k, img in enumerate(sphere_examples(bsdf)):
      img_shape = img.shape
      img = img.squeeze(-2).clamp(0,1).detach().cpu().numpy()
      plt.imsave(f"outputs/base_{k:02}.png", img)
      axes[unroll(k, c)].imshow(img)
      axes[unroll(k, c)].axis('off')
    plt.savefig(f"outputs/bases.png", bbox_inches="tight")
    plt.clf()
    plt.close(f)

def test():
  with torch.no_grad():
    for i in trange(len(perspectives)):
      R, T = perspectives[i]
      cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
      if isinstance(bsdf, ComposeSpatialVarying):
        got, _ = pt.pathtrace(
          shape,
          size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=BasisBRDF(bsdf),
          cameras=cameras, lights=lights, device=device, silent=True,
          background=0,
        )
        f, axes = plt.subplots(r, c)
        f.set_figheight(10)
        f.set_figwidth(10)
        got = got.unsqueeze(-1).expand(got.shape + (3,))
        for k, img in enumerate(got.split(1, dim=-2)):
          img = img.squeeze(-2).cpu().numpy()
          axes[unroll(k, c)].imshow(img)
          axes[unroll(k, c)].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"outputs/weights_{i:04}.png", bbox_inches="tight")
        plt.clf()
        plt.close(f)
      #normals, _ = pt.pathtrace(
      #  shape,
      #  size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=Debug(),
      #  cameras=cameras, lights=lights, device=device, silent=True,
      #  background=0,
      #)
      #save_image(f"outputs/normals_{i:04}.png", normals)
      #illum, _ = pt.pathtrace(
      #  shape,
      #  size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=Illumination(),
      #  cameras=cameras, lights=lights, device=device, silent=True,
      #)
      #save_image(f"outputs/illum_{i:04}.png", illum)


      if (integrator is not None) and False:
        got, _ = pt.pathtrace(
          shape,
          size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=integrator,
          cameras=cameras, lights=lights, device=device, silent=True,
          background=0,
        )
        save_image(f"outputs/got_{i:04}.png", got)

test()

