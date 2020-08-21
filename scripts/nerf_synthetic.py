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
  LossSampler, masked_loss, rand_uv, load_image,
  eikonal_loss,
)
from pytorch3d.pathtracer.training_utils import (
  save_image, save_plot, train_nerf, test_nerf, test_nerf_resources,
)
from pytorch3d.pathtracer.lights import ( LightField )

# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform, HardPhongShader,
    look_at_rotation,
    MeshRasterizer, MeshRenderer, RasterizationSettings,
)
import json
from tqdm import trange, tqdm

SIZE=256
dataset="materials"
DIR=f"nerf_synthetic/{dataset}/"
iters = 25_000
print(f"{dataset}, Size: {SIZE}, Iters: {iters}")

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

tfs = json.load(open(DIR + "transforms_train.json"))
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

integrator=Direct()

shape = torch.jit.load(f"models/{dataset}_sdf_f.pt", device)
density_field = SDF(sdf=shape)
#density_field = SDF(sdf=torch.jit.script(SphereSDF(n=2<<6)))

density_field.max_steps=64

#learned_bsdf = torch.load(f"models/{dataset}_bsdf_f.pt")
learned_bsdf = ComposeSpatialVarying([
  NeuralBSDF(activation=nn.Softplus()) for _ in range(8)
])

#lights = torch.load(f"models/{dataset}_light_f.pt")
lights = LightField()

surface_lr = 8e-5
bsdf_lr    = 8e-4
light_lr   = 8e-5
print(f"Learning rate is S: {surface_lr}, B: {bsdf_lr}, L: {light_lr}")
opt = torch.optim.AdamW([
  { 'params': density_field.parameters(), 'lr':surface_lr, },
  { 'params': learned_bsdf.parameters(),  'lr':bsdf_lr, },
  { 'params': lights.parameters(),        'lr':light_lr, },
], lr=surface_lr, weight_decay=0)
def extra_loss(mi, got, exp, mask):
  # might need to add in something for eikonal loss over all space
  raw_n = getattr(mi,  "raw_normals", None)
  if raw_n is None: return 0
  return eikonal_loss(raw_n)
losses = train_nerf(
  density_field,
  bsdf=learned_bsdf,
  integrator=integrator,
  lights=lights,
  focal=focal,
  cam_to_worlds=cam_to_worlds,
  exp_imgs=exp_imgs,
  exp_masks=exp_masks,
  opt=opt,
  size=SIZE,

  crop_size=80,

  save_freq=5000,
  valid_freq=1000,
  max_valid_size=SIZE,
  iters=iters,
  N=6,
  extra_loss=extra_loss,

  name_fn=lambda i: f"outputs/train_{dataset}_{i:06}.png",
  valid_name_fn=lambda i: f"outputs/valid_{dataset}_{i:06}.png",
  silent=True,
  uv_select=lambda _, crop_size: rand_uv(SIZE, SIZE, crop_size),
)

if iters > 0:
  torch.jit.save(density_field.sdf, f"models/{dataset}_sdf_f.pt")
  torch.save(learned_bsdf, f"models/{dataset}_bsdf_f.pt")
  torch.save(lights, f"models/{dataset}_light_f.pt")

density_field.max_steps=256

cam_to_worlds, focal, exp_imgs, exp_masks = test_nerf_resources(DIR, SIZE)

print("Running on test set")

test_nerf(
  density_field,
  integrator=integrator,
  bsdf=learned_bsdf,
  lights=lights,
  cam_to_worlds=cam_to_worlds,
  focal=focal,
  exp_imgs=exp_imgs,
  size=SIZE,

  name_fn=lambda i: f"outputs/test_{dataset}_{i:03}.png",
)

