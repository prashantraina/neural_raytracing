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
  Mask, Debug, Path, Depth, NeRFIntegrator,
  NeuralApprox, Direct,
)
from pytorch3d.pathtracer.shapes.sdfs import (
  SDF, SphereSDF, CapsuleSDF, RoundBoxSDF,
)
#from pytorch3d.pathtracer.neural_blocks import ( NormalMLP )
from pytorch3d.pathtracer.utils import (
  LossSampler, masked_loss, rand_uv, load_image,
  eikonal_loss,
)
from pytorch3d.pathtracer.training_utils import ( save_image, save_plot, train_dtu, test_dtu )
from pytorch3d.pathtracer.lights import ( PointLights, LightField, )
from pytorch3d.pathtracer.cameras import ( NeRFCamera )

# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform, HardPhongShader,
    look_at_rotation,
    MeshRasterizer, MeshRenderer, RasterizationSettings,
)
import json
from tqdm import trange, tqdm

SIZE=256
DIR="DTU/"
iters = 25_000
dataset = "69"
DIR = os.path.join(DIR, f"scan{dataset}")
print(f"DTU, Size: {SIZE}, Iters: {iters}, Scan: {dataset}")

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
density_field = SDF(sdf=shape)
#density_field = SDF(sdf=torch.jit.script(SphereSDF(n=2<<5)))

density_field.max_steps = 64


learned_bsdf = torch.load(f"models/dtu_{dataset}_bsdf.pt")
for bsdf in learned_bsdf.bsdfs: setattr(bsdf, "act", nn.Sigmoid())
#learned_bsdf = ComposeSpatialVarying([
#  NeuralBSDF() for _ in range(10)
#] + [
#  Diffuse(preprocess=torch.sigmoid).random() for _ in range(6)
#])

lights = torch.load(f"models/dtu_{dataset}_lights.pt")
#lights = LightField()

torch.jit.save(density_field.sdf, "models/tmp.pt")
torch.save(learned_bsdf, "models/tmp.pt")
torch.save(lights, "models/tmp.pt")

test_idxs = list(range(0, len(poses), 10))
train_poses = [pose for i, pose in enumerate(poses) if i not in test_idxs]
train_intrinsics = torch.stack(
  [intr for i, intr in enumerate(intrinsics) if i not in test_idxs], dim=0
)
train_e_imgs = [img for i, img in enumerate(exp_imgs) if i not in test_idxs]
train_e_masks = [mask for i, mask in enumerate(exp_masks) if i not in test_idxs]

surface_lr= 8e-5
bsdf_lr   = 8e-5
light_lr  = 8e-5
print(f"LR rate is S: {surface_lr}, B: {bsdf_lr}, L: {light_lr}")
opt = torch.optim.AdamW([
  { 'params': density_field.parameters(), 'lr':surface_lr },
  { 'params': learned_bsdf.parameters(),  'lr':bsdf_lr, },
  { 'params': lights.parameters(),        'lr':light_lr, },
], lr=surface_lr, weight_decay=0)
def extra_loss(mi, got, exp, mask):
  # might need to add in something for eikonal loss over all space
  raw_n = getattr(mi,  "raw_normals", None)
  loss = 0
  if raw_n is not None: loss = loss + eikonal_loss(raw_n)

  return loss
losses = train_dtu(
  density_field, bsdf=learned_bsdf, integrator=integrator,
  lights=lights, poses=train_poses, intrinsics=train_intrinsics,

  exp_imgs=train_e_imgs, exp_masks=train_e_masks,
  opt=opt,
  size=SIZE, crop_size=96,
  save_freq=5000, valid_freq=1000,
  max_valid_size=128,
  N=4,
  iters=iters,
  extra_loss=extra_loss,
  uv_select=lambda _, crop_size: rand_uv(SIZE,SIZE,crop_size),
  silent=True,

  name_fn=lambda i: f"outputs/train_dtu_{dataset}_{i:06}.png",
  valid_name_fn=lambda i: f"outputs/valid_dtu_{dataset}_{i:06}.png",
)

if iters > 0:
  torch.jit.save(
    density_field.sdf,
    f"models/dtu_{dataset}_sdf.pt"
  )
  torch.save(
    learned_bsdf,
    f"models/dtu_{dataset}_bsdf.pt"
  )
  torch.save(
    lights,
    f"models/dtu_{dataset}_lights.pt"
  )

test_poses = [pose for i, pose in enumerate(poses) if i in test_idxs]
test_intrinsics = torch.stack(
  [intr for i, intr in enumerate(intrinsics) if i in test_idxs], dim=0
)
test_e_imgs = [img for i, img in enumerate(exp_imgs) if i in test_idxs]
test_e_masks = [mask for i, mask in enumerate(exp_masks) if i in test_idxs]

test_dtu(
  density_field, integrator=integrator, bsdf=learned_bsdf, lights=lights,
  poses=test_poses, intrinsics=test_intrinsics,

  exp_imgs=test_e_imgs, exp_masks=test_e_masks, size=SIZE,
  name_fn=lambda i: f"outputs/test_dtu_{dataset}_{i:03}.png",
)
