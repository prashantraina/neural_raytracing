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
from pytorch3d.pathtracer.integrators import ( Direct )
from pytorch3d.pathtracer.shapes.sdfs import (
  SDF, SphereSDF, CapsuleSDF, RoundBoxSDF,
)
from pytorch3d.pathtracer.utils import (
  LossSampler, masked_loss, rand_uv, eikonal_loss,
)
from pytorch3d.pathtracer.training_utils import (
  save_image, save_plot, train_nerv_ptl, test_nerv_ptl,
)
from pytorch3d.pathtracer.neural_blocks import ( SkipConnMLP )
import imageio
import json
import cv2
import imageio

SIZE=200
dataset = "armadillo"
assert(dataset in ["armadillo", "hotdogs"])
DIR = f"nerv_public_release/{dataset}/"
iters = 75_000
var = "_sigmoid"
assert(var in ["", "_clamp", "_sigmoid"])
with_norm = var != "_sigmoid"

print(f"NeRV({dataset}{var}), size({SIZE}), iters({iters})")

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def load_image(path): return torch.from_numpy(imageio.imread(path))

tfs = json.load(open(DIR + "train_point/transforms_train.json"))
exp_imgs = []
exp_masks = []
light_locs = []
focal = 0.5 * SIZE / np.tan(0.5 * float(tfs['camera_angle_x']))
cam_to_worlds=[]
for frame in tfs["frames"]:
  img = load_image(os.path.join(DIR, "train_point", frame['file_path'] + '.exr')).to(device)
  assert(img.min() >= 0)
  exp_imgs.append(img[..., :3])
  exp_masks.append((img[..., 3] - 1e-5).ceil())
  tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
  # set distance to 1 from origin
  n = torch.linalg.norm(tf_mat[:3, 3], dim=-1)
  if with_norm: tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
  cam_to_worlds.append(tf_mat)
  # also have to update light positions since normalizing to unit sphere
  ll = torch.tensor(frame['light_loc'], dtype=torch.float, device=device)
  if with_norm:
    ln = torch.linalg.norm(ll, dim=-1)
    light_locs.append(ln/n * F.normalize(ll, dim=-1))
  else:
    light_locs.append(ll)

integrator=Direct(training=True)

load = True
if load:
  shape = torch.jit.load(f"models/nerv_{dataset}{var}_sdf.pt")
  density_field = SDF(sdf=shape)
  density_field.max_steps = 64
  density_field.dist = 2.2 if with_norm else 8

  learned_bsdf =torch.load(f"models/nerv_{dataset}{var}_bsdf.pt")

  occ_mlp = torch.load(f"models/nerv_{dataset}{var}_occ.pt")
else:
  density_field = SDF(sdf=torch.jit.script(SphereSDF(n=2<<6)), dist=2.2 if with_norm else 8)
  density_field.max_steps = 64
  learned_bsdf = ComposeSpatialVarying([
    NeuralBSDF(activation=nn.Softplus()) for _ in range(7)
  ])
  occ_mlp = SkipConnMLP(
    in_size=5, out=1,
    device=device,
  ).to(device)


torch.jit.save(density_field.sdf, f"models/tmp.pt")
torch.save(learned_bsdf, f"models/tmp.pt")
torch.save(occ_mlp, f"models/tmp.pt")

surface_lr   = 4e-5
bsdf_lr      = 4e-5
occ_lr       = 4e-5
print(f"Learning rate is {surface_lr}, {bsdf_lr}, {occ_lr}")
opt = torch.optim.AdamW([
  { 'params': density_field.parameters(), 'lr':surface_lr,},
  { 'params': learned_bsdf.parameters(),  'lr':bsdf_lr, },
  { 'params': occ_mlp.parameters(),  'lr': occ_lr, },
], lr=surface_lr, weight_decay=0)

def extra_loss(mi, got, exp, mask):
  # might need to add in something for eikonal loss over all space
  raw_n = getattr(mi,  "raw_normals", None)
  loss = 0
  if raw_n is not None: loss = loss + eikonal_loss(raw_n)

  return loss

def save_fn(_=None):
  if iters > 0:
    print("Saving...", dataset, var)
    torch.jit.save(
      density_field.sdf, f"models/nerv_{dataset}{var}_sdf.pt"
    )
    torch.save(
      learned_bsdf, f"models/nerv_{dataset}{var}_bsdf.pt"
    )
    torch.save(
      occ_mlp, f"models/nerv_{dataset}{var}_occ.pt"
    )

losses = train_nerv_ptl(
  density_field,
  bsdf=learned_bsdf,
  integrator=integrator,
  light_locs=light_locs,
  focal=focal, cam_to_worlds=cam_to_worlds,
  exp_imgs=exp_imgs, exp_masks=exp_masks,
  opt=opt, size=SIZE,
  crop_size=64,
  save_freq=20_000,
  valid_freq=2000,
  max_valid_size=100,
  iters=iters,
  N=6,
  extra_loss=extra_loss,
  save_fn=save_fn,
  uv_select=lambda _, crop_size: rand_uv(SIZE,SIZE,crop_size),
  w_isect = occ_mlp,
)

save_fn()

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
  if with_norm: tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
  cam_to_worlds.append(tf_mat)
  # also have to update light positions since normalizing to unit sphere
  ll = torch.tensor(frame['light_loc'], dtype=torch.float, device=device)
  if with_norm:
    ln = torch.linalg.norm(ll, dim=-1)
    light_locs.append(ln/n * F.normalize(ll, dim=-1))
  else:
    light_locs.append(ll)

print("Occ-MLP variant:")

density_field.max_steps = 128

test_nerv_ptl(
  density_field, bsdf=learned_bsdf, integrator=integrator, light_locs=light_locs,
  exp_imgs=exp_imgs,
  cam_to_worlds=cam_to_worlds, focal=focal, size=SIZE,

  name_fn=lambda i: f"outputs/test_nerv_{dataset}{var}_occ_{i:03}.png",
  w_isect=occ_mlp,
)

# Hard shadow variant

print("Hard shadow variant:")

test_nerv_ptl(
  density_field, bsdf=learned_bsdf, integrator=integrator, light_locs=light_locs,
  exp_imgs=exp_imgs,
  cam_to_worlds=cam_to_worlds, focal=focal, size=SIZE,

  name_fn=lambda i:f"outputs/test_nerv_{dataset}{var}_hs_{i:03}.png",
  w_isect=True,
)
