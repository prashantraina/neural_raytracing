import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2

import pytorch3d.pathtracer as pt
from pytorch3d.pathtracer.bsdf import (
  Diffuse, ComposeSpatialVarying, Phong, Plastic, NeuralBSDF, Bidirectional,
)
from pytorch3d.pathtracer.integrators import ( Direct, Debug, Luminance, Illumination )
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
import json
from tqdm import trange, tqdm
import random

SIZE=256
DIR="DTU/"
dataset = "110"
DIR = os.path.join(DIR, f"scan{dataset}")
print(f"Editting DTU, Size: {SIZE}, Scan: {dataset}")

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

# hyperparameters


integrator=Debug()

shape = torch.jit.load(f"models/dtu_{dataset}_sdf.pt", device)

k=-10
def bend(p):
  x,y,z = p.split(1, dim=-1)
  v = z * k
  c = v.cos()
  s = v.sin()
  warped = torch.cat([
    c * x - s * z,
    y,
    s * x + c * z,
  ],dim=-1)
  return shape(warped)
def disp(p):
  x,y,z = p.split(1, dim=-1)
  out = shape(p)
  return out + 0.05*((20 * x).cos()*(20*y).cos()*(20*z).cos()).reshape_as(out)
#density_field = SDF(sdf=bend)
density_field = SDF(sdf=shape)

density_field.max_steps = 128

# inverse of above
def preproc_inv(p):
  x,y,z = p.split(1, dim=-1)
  v = z * k
  c = v.cos()
  s = -v.sin()
  warped = torch.cat([
    c * x - s * z,
    y,
    s * x + c * z,
  ],dim=-1)
  return warped

learned_bsdf = torch.load(f"models/dtu_{dataset}_bsdf.pt")
#learned_bsdf.preprocess = preproc_inv

#choices = [
#  Diffuse([19., 8, 0]),
#  Diffuse([0., 13, 16]),
#  learned_bsdf,
#]
#class Noise(nn.Module):
#  def __init__(self): super().__init__()
#  def forward(self, p):
#    def noise(v):
#      s = (v - v.floor())
#      return 100 * (s * 13.48).sin()
#    x,y,z = p.split(1, dim=-1)
#    return torch.cat([
#      noise(x * z) * noise(x * y),
#      noise(y * z) * noise(x * y),
#      noise(x * z) * noise(z * y)+50,
#    ], dim=-1)
#learned_bsdf = ComposeSpatialVarying(choices)
#learned_bsdf.sp_var_fn = Noise()

lights = torch.load(f"models/dtu_{dataset}_lights.pt")
#setattr(lights, 'preproc', preproc_inv)
#setattr(lights, 'postproc', lambda x: x.clamp(min=0.3))
#setattr(lights, 'preproc', lambda x: x)

test_dtu(
  density_field, integrator=integrator, bsdf=learned_bsdf, lights=lights,
  poses=poses, intrinsics=intrinsics,

  exp_imgs=exp_imgs, exp_masks=exp_masks, size=SIZE,
  name_fn=lambda i: f"outputs/edit_dtu_{dataset}_{i:03}.png",
)
