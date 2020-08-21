import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from pytorch3d.io import load_objs_as_meshes
import pytorch3d.pathtracer as pt
from pytorch3d.pathtracer.bsdf import (
  Diffuse, ComposeSpatialVarying, Phong, Plastic, NeuralBSDF,
)
from pytorch3d.pathtracer.integrators import (
  Mask, Debug, Path, Depth, NeRFIntegrator,
  NeuralApprox, Direct, BasisBRDF,
)
from pytorch3d.pathtracer.cameras import ( NeRFCamera )
from pytorch3d.pathtracer.lights import ( PointLights )
from pytorch3d.pathtracer.shapes.sdfs import (
  SDF, SphereSDF, CapsuleSDF, RoundBoxSDF,
)
from pytorch3d.pathtracer.utils import (
  rand_uv, eikonal_loss, count_parameters, sphere_examples,
)
from pytorch3d.pathtracer.training_utils import (
  save_image, save_plot, train_nerv_ptl, test_nerv_ptl,
)
from pytorch3d.pathtracer.neural_blocks import ( SkipConnMLP )
import imageio
import json
import math
from tqdm import tqdm

SIZE=200
dataset = "armadillo"
assert(dataset in ["armadillo", "hotdogs"])
DIR = f"nerv_public_release/{dataset}/"

print(f"visualize NeRV({dataset}), size({SIZE})")

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def load_image(path): return torch.from_numpy(imageio.imread(path))

integrator=Direct()

shape = torch.jit.load(f"models/nerv_{dataset}_sdf.pt")
shape = SDF(sdf=shape)
print("Number of shape parameters:", count_parameters(shape.parameters()))
shape.max_steps = 64

bsdf = torch.load(f"models/nerv_{dataset}_bsdf.pt")
print("Number of bsdf parameters:", count_parameters(bsdf.parameters()))

occ_mlp = torch.load(f"models/nerv_{dataset}_occ.pt")
print("Number of occ parameters:", count_parameters(occ_mlp.parameters()))

def naive_squarest_divisor(v):
  sqrt = math.sqrt(v)
  curr = 1
  for i in range(2, v+1):
    if (v % i == 0) and (abs(sqrt-i) < abs(sqrt - curr)): curr = i
  return curr, v//curr

def unroll(i, w): return i//w, i % w

if isinstance(bsdf, ComposeSpatialVarying):
  N = len(bsdf.bsdfs)
  r, c = naive_squarest_divisor(N)
  print("Using axes", r, c)
  assert(r * c == N)
  with torch.no_grad():
    if r == 1: unroll = lambda v, _: v
    f, axes = plt.subplots(r, c)
    for k, img in enumerate(sphere_examples(bsdf)):
      img_shape = img.shape
      img = img.squeeze(-2).clamp(0,1).detach().cpu().numpy() ** (1/2.2)
      plt.imsave(f"outputs/nerv_base_{k:02}.png", img)
      axes[unroll(k, c)].imshow(img)
      axes[unroll(k, c)].axis('off')
    plt.savefig(f"outputs/nerv_bases.png", bbox_inches="tight")
    plt.clf()
    plt.close(f)

tfs = json.load(open(DIR + "transforms_test.json"))
exp_imgs = []
exp_masks = []
light_locs = []
focal = 0.5 * SIZE / np.tan(0.5 * float(tfs['camera_angle_x']))
cam_to_worlds=[]
for frame in tfs["frames"][:100]:
  img = load_image(os.path.join(DIR, frame['file_path'] + '.exr')).to(device)
  exp_imgs.append(img[..., :3])
  exp_masks.append((img[..., 3] - 1e-5).ceil())
  tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
  # set distance to 1 from origin
  n = torch.linalg.norm(tf_mat[:3, 3], dim=-1)
  tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
  cam_to_worlds.append(tf_mat)
  # also have to update light positions since normalizing to unit sphere
  ll = torch.tensor(frame['light_loc'], dtype=torch.float, device=device)
  ln = torch.linalg.norm(ll, dim=-1)
  light_locs.append(ln/n * F.normalize(ll, dim=-1))


def test():
  with torch.no_grad():
    for i, (c2w, lp) in enumerate(zip(tqdm(cam_to_worlds), light_locs)):
      exp = exp_imgs[i].clamp(min=0, max=1)
      cameras = NeRFCamera(cam_to_world=c2w.unsqueeze(0), focal=focal, device=device)
      lights = PointLights(intensity=[1,1,1], location=lp[None,...], scale=100, device=device)

      if isinstance(bsdf, ComposeSpatialVarying):
        got = pt.pathtrace(
          shape,
          size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=BasisBRDF(bsdf),
          cameras=cameras, lights=lights, device=device, silent=True,
        )[0].clamp(min=0, max=1)
        f, axes = plt.subplots(r, c)
        f.set_figheight(10)
        f.set_figwidth(10)
        got = got.unsqueeze(-1).expand(got.shape + (3,))
        wm_0 = None
        wm_1 = None
        for k, img in enumerate(got.split(1, dim=-2)):
          img = img.squeeze(-2).cpu().numpy()
          axes[unroll(k, c)].imshow(img)
          axes[unroll(k, c)].axis('off')
          if k == 0: wm_0 = img
          if k == 1: wm_1 = img
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"outputs/nerv_weights_{i:04}.png", bbox_inches="tight")
        plt.clf()
        plt.close(f)

        # render first two and normalize for easy figure
        f, axes = plt.subplots(2)
        f.set_figheight(10)
        f.set_figwidth(10)
        total = wm_0 + wm_1
        wm_0 = wm_0/total
        wm_1 = wm_1/total
        axes[0].imshow(wm_0)
        axes[0].axis('off')
        axes[1].imshow(wm_1)
        axes[1].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"outputs/nerv_wm01_{i:04}.png", bbox_inches="tight")
        plt.clf()
        plt.close(f)
      normals = pt.pathtrace(
        shape,
        size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=Debug(),
        cameras=cameras, lights=lights, device=device, silent=True,
      )[0]
      save_image(f"outputs/nerv_normals_{i:04}.png", normals)

      if (integrator is not None) and False:
        got = pt.pathtrace(
          shape,
          size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=integrator,
          cameras=cameras, lights=lights, device=device, silent=True,
        )[0].clamp(min=0, max=1)
        save_image(f"outputs/got_{i:04}.png", got)


test()

