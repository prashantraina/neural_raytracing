import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import pytorch3d.pathtracer as pt
from pytorch3d.pathtracer.bsdf import (
  Diffuse, ComposeSpatialVarying, Phong, Plastic, NeuralBSDF, Conductor,
)
from pytorch3d.pathtracer.integrators import (
  Mask, Debug, Path, Depth, NeRFIntegrator, NeuralApprox, Direct,
)
from pytorch3d.pathtracer.shapes.sdfs import (
  SDF, SphereSDF, CapsuleSDF, RoundBoxSDF,
)
from pytorch3d.pathtracer.utils import (
  LossSampler, masked_loss, rand_uv, load_image, eikonal_loss,
)
from pytorch3d.pathtracer.training_utils import ( save_image, save_plot, train_nerf, test_nerf )
from pytorch3d.pathtracer.lights import (
  PointLights,
  LightField,
)

import json
import random

SIZE=256
dataset="drums"
DIR=f"nerf_synthetic/{dataset}/"
print("Testing NeRF", dataset)

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

tfs = json.load(open(DIR + "transforms_test.json"))
exp_imgs = []
exp_masks = []
focal = 0.5 * SIZE / np.tan(0.5 * float(tfs['camera_angle_x']))
cam_to_worlds=[]
with torch.no_grad():
  for frame in tfs["frames"]:
    img = load_image(os.path.join(DIR, frame['file_path'] + '.png'), resize=(SIZE, SIZE))\
      .to(device)
    exp_imgs.append(img[..., :3])
    exp_masks.append((img[..., 3] - 1e-5).ceil())
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)

integrator=Debug()

shape = torch.jit.load(f"models/{dataset}_sdf_f.pt", device)
k=2.5
def bend(p):
  x,y,z = p.split(1, dim=-1)
  v = y * k
  c = v.cos()
  s = -v.sin()
  warped = torch.cat([
    c * x - s * z,
    y,
    s * x + c * z,
  ],dim=-1)
  return shape(warped)
a = 0.1
def trans(p):
  warped = a * (p[..., 1] > - 0.05)
  out = p.clone()
  out[..., 0] = out[..., 0] + warped
  return shape(out)
def inv_trans(p):
  warped = a * (p[..., 1] > - 0.05)
  out = p.clone()
  out[..., 0] = out[..., 0] + warped
  return p

def add_hole(p):
  prev = shape(p)
  intersect = torch.linalg.norm(p, dim=-1, keepdim=True) - 0.2
  # take -intersect.reshape_as(prev) for subtraction
  return torch.maximum(prev, intersect.reshape_as(prev))
s=0.2
def scale_z(p):
  x,y,z = p.split(1, dim=-1)
  warped = torch.cat([
    x,
    y,
    z / s,
  ],dim=-1)
  return shape(warped)
def inv_scale_z(p):
  x,y,z = p.split(1, dim=-1)
  warped = torch.cat([
    x,
    y,
    z / s,
  ],dim=-1)
  return warped


density_field = SDF(sdf=shape)
density_field.max_steps=64

learned_bsdf = torch.load(f"models/{dataset}_bsdf_f.pt")

# inverse of above
def bend_inv(p):
  x,y,z = p.split(1, dim=-1)
  v = y * k
  c = v.cos()
  s = v.sin()
  warped = torch.cat([
    c * x - s * z,
    y,
    s * x + c * z,
  ],dim=-1)
  return warped
#learned_bsdf.preprocess = inv_trans

#class Split(nn.Module):
#  def __init__(self): super().__init__()
#  def forward(self, p):
#    out_shape = p.shape[:-1] + (2,)
#    #thresh = p[..., 1] > -0.05
#    thresh = ((p[..., 1] * 25).cos()) > 0
#    out = torch.where(
#      thresh.unsqueeze(-1),
#      torch.tensor([1000, 0], device=p.device, dtype=torch.float).expand(out_shape),
#      torch.tensor([0, 1000], device=p.device, dtype=torch.float).expand(out_shape),
#    )
#    return out
#learned_bsdf = ComposeSpatialVarying([
#  learned_bsdf,
#  Diffuse([6.6,2.2,1.2]),
#])
#learned_bsdf.sp_var_fn = Split()

# flipping axis experiment
def flip_axis(p):
  p[...,1] = -p[...,1]
  return p
def flip_axis(p):
  p[..., 1] = p[...,1]-1

  p[..., 1][p[..., 1] < -1] = 2 + p[...,1][p[...,1] < -1]
  return p
#learned_bsdf.preprocess = flip_axis


# Make experiment even
#choices = [
#  Conductor().random(),
#  Diffuse([0.66,0.22,0.12]),
#]
#class Even(nn.Module):
#  def __init__(self): super().__init__()
#  def forward(self, p):
#    return torch.zeros(p.shape[:-1] + (2,), device=p.device, dtype=torch.float)
#learned_bsdf= ComposeSpatialVarying(choices)
#learned_bsdf.sp_var_fn = Even()

#random.shuffle(learned_bsdf.bsdfs)
#for i in range(len(learned_bsdf.bsdfs)):
#  if random.random() > 0.4:
#    learned_bsdf.bsdfs[i] = Diffuse([0.,0.,0.], device=device)
  #if random.random() > 0.4:
  #  learned_bsdf.bsdfs[i] = random.choice(choices)

lights = torch.load(f"models/{dataset}_light_f.pt")
#setattr(lights, 'preproc', inv_trans)
#setattr(lights, 'postproc', lambda x: x.clamp(min=1e-2))

losses = test_nerf(
  density_field,
  integrator=integrator,
  bsdf=learned_bsdf,
  lights=lights,
  cam_to_worlds=cam_to_worlds,
  focal=focal,
  exp_imgs=exp_imgs,
  size=SIZE,

  name_fn=lambda i: f"outputs/testset_{dataset}_{i:03}.png",
)

