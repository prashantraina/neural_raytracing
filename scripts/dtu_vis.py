import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

from pytorch3d.io import load_objs_as_meshes
import pytorch3d.pathtracer as pt
from pytorch3d.pathtracer.bsdf import (
  Diffuse, ComposeSpatialVarying, Phong, Plastic, NeuralBSDF,
)
from pytorch3d.pathtracer.integrators import (
  Mask, Debug, Path, Depth, NeRFIntegrator, NeuralApprox, Direct, BasisBRDF,
)
from pytorch3d.pathtracer.shapes.sdfs import ( SDF, SphereSDF )
from pytorch3d.pathtracer.utils import (
  load_image, sphere_examples,
)
from pytorch3d.pathtracer.training_utils import ( save_image, save_plot )
from pytorch3d.pathtracer.lights import ( PointLights, )
from pytorch3d.pathtracer.cameras import ( DTUCamera )
import json
from tqdm import tqdm

SIZE=256
DIR="DTU/"
dataset = "55"
DIR = os.path.join(DIR, f"scan{dataset}")
print(f"visualize DTU, Size: {SIZE}, Scan: {dataset}")

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

num_imgs = 0
exp_imgs = []
exp_masks = []
mask_dir = os.path.join(DIR, "mask")
for f in sorted(os.listdir(mask_dir)):
  if f.startswith("._"): continue
  mask = load_image(os.path.join(mask_dir, f), resize=(SIZE, SIZE)).to(device)
  num_imgs += 1
  exp_masks.append(mask.max(dim=-1)[0].ceil())

image_dir = os.path.join(DIR, "image")
for f in sorted(os.listdir(image_dir)):
  if f.startswith("._"): continue
  img = load_image(os.path.join(image_dir, f), resize=(SIZE, SIZE)).to(device)
  exp_imgs.append(img)

assert(len(exp_imgs) == len(exp_masks))

tfs = np.load(os.path.join(DIR, "cameras.npz"))
Ps = [tfs[f"world_mat_{i}"] @ tfs[f"scale_mat_{i}"]  for i in range(num_imgs)]
def KRt_from_P(P):
  K, R, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
  K = K/K[2,2]
  intrinsics = np.eye(4)
  intrinsics[:3, :3] = K

  pose = np.eye(4, dtype=np.float32)
  pose[:3, :3] = R.transpose()
  pose[:3,3] = (t[:3] / t[3])[:,0]
  return torch.from_numpy(intrinsics).float().cuda(), torch.from_numpy(pose).float().cuda()

intrinsics, poses = list(zip(*[
  KRt_from_P(p[:3, :4]) for p in Ps
]))
poses = torch.stack(poses, dim=0)
# normalize distance to be at most 1 for convenience
max_dist = torch.linalg.norm(poses[:, :3, 3], dim=-1).max()
poses[:, :3, 3] /= max_dist
intrinsics = torch.stack(intrinsics, dim=0)

integrator=Direct()

shape = torch.jit.load(f"models/dtu_{dataset}_sdf.pt", device)
shape = SDF(sdf=shape)

shape.max_steps = 64


bsdf = torch.load(f"models/dtu_{dataset}_bsdf.pt")

lights = torch.load(f"models/dtu_{dataset}_lights.pt")

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
      img = img.squeeze(-2).clamp(0,1).detach().cpu().numpy()
      plt.imsave(f"outputs/base_{k:02}.png", img)
      axes[unroll(k, c)].imshow(img)
      axes[unroll(k, c)].axis('off')
    plt.savefig(f"outputs/bases.png", bbox_inches="tight")
    plt.clf()
    plt.close(f)

def test():
  with torch.no_grad():
    for i, (pose, intrinsic) in enumerate(zip(tqdm(poses), intrinsics)):
      cameras = DTUCamera(pose=pose[None, ...], intrinsic=intrinsic[None, ...], device=device)
      if isinstance(bsdf, ComposeSpatialVarying):
        got, _ = pt.pathtrace(
          shape,
          size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=BasisBRDF(bsdf),
          cameras=cameras, lights=lights, device=device, silent=True,
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
      normals, _ = pt.pathtrace(
        shape,
        size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=Debug(),
        cameras=cameras, lights=lights, device=device, silent=True,
        background=1,
      )
      save_image(f"outputs/normals_{i:04}.png", normals)

      if (integrator is not None):
        got = pt.pathtrace(
          shape,
          size=SIZE, chunk_size=SIZE, bundle_size=1, bsdf=bsdf, integrator=integrator,
          cameras=cameras, lights=lights, device=device, silent=True,
          background=1,
        )[0].clamp(min=0, max=1)
        save_image(f"outputs/got_{i:04}.png", got)

test()

