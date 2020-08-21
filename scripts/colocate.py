import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from pytorch3d.io import load_objs_as_meshes
import pytorch3d.pathtracer as pt
from pytorch3d.pathtracer.bsdf import (
  Diffuse, ComposeSpatialVarying, Phong, Plastic, NeuralBSDF, Conductor,
)
from pytorch3d.pathtracer.integrators import (
  Mask, Debug, Path, Depth, NeRFIntegrator,
  NeuralApprox, Direct,
)
from pytorch3d.pathtracer.shapes.sdfs import ( SDF, SphereSDF )
from pytorch3d.pathtracer.training_utils import (
  train_sample, save_image, test,
  test_colocate_resources,
)
from pytorch3d.pathtracer.utils import (
  LossSampler, masked_loss, rand_uv, depth_image, eikonal_loss,
  load_image,
)
from pytorch3d.pathtracer.lights import ( LightField, PointLights )
from pytorch3d.pathtracer.neural_blocks import ( SkipConnMLP )

# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform, OpenGLPerspectiveCameras, HardPhongShader,
    look_at_rotation,
)
from tqdm import trange, tqdm
from itertools import chain

SIZE=256
N_VIEWS=8
DIST=1
iters = int(50_000)


print(f"Colocate light, Iters: {iters}")

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def train_on_kind(k):
  Rs = []; Ts = []
  exp_imgs = []
  exp_masks = []
  for i, elev in enumerate(torch.linspace(0, 45, N_VIEWS, device=device)):
    for j, azim in enumerate(torch.linspace(-90, 90, N_VIEWS, device=device)):
      R, T = look_at_view_transform(dist=DIST, elev=elev, azim=azim, device=device)
      Rs.append(R); Ts.append(T)
      img = load_image(f"mitsuba_scenes/cbox_relight/{k}_{i:03}_{j:03}.png", (SIZE, SIZE)).to(device)
      exp_imgs.append(img[..., :3])
      exp_masks.append(img[..., 3])

  if False:
    density_field = SDF(sdf=torch.jit.script(SphereSDF(n=2<<5)))
  else:
    sdf = torch.jit.load(f"models/col_{k}_sdf.pt")
    density_field = SDF(sdf=sdf)
    density_field.max_steps = 64

  if True:
    learned_bsdf = ComposeSpatialVarying([
      *[NeuralBSDF() for _ in range(2)],
      Diffuse(preprocess=nn.Softplus()).random(),
      Conductor(activation=nn.Softplus(), device=device).random(),
    ])
  else:
    learned_bsdf = torch.load(f"models/col_{k}_bsdf.pt")

  integrator = Direct()

  lights = PointLights(device=device, scale=5)

  occ_mlp = SkipConnMLP(
    in_size=5, out=1,
    device=device,
  ).to(device)

  surface_lr = 8e-5
  bsdf_lr    = 8e-5
  light_lr   = 8e-5
  print(f"Surface LR for {k} is {surface_lr}, BSDF LR is {bsdf_lr}, L LR is {light_lr}")
  opt = torch.optim.AdamW([
    { 'params': density_field.parameters(), 'lr':surface_lr, },
    { 'params': learned_bsdf.parameters(), 'lr':bsdf_lr, },
    { 'params': lights.intensity_parameters(), 'lr': light_lr, },
    { 'params': occ_mlp.parameters(), 'lr': 8e-5, },
  ], lr=surface_lr, weight_decay=0)

  def extra_loss(mi, got, exp, mask):
    # might need to add in something for eikonal loss over all space
    raw_n = getattr(mi,  "raw_normals", None)
    loss = 0
    if raw_n is not None: loss = loss + eikonal_loss(raw_n)

    raw_w = getattr(mi, 'normalized_weights', None)
    if raw_w is not None: loss = loss + 1e-2 * raw_w.std(dim=-1).mean()

    return loss

  def light_update(cam, light): light.location = cam.get_camera_center()* 1.05

  losses = train_sample(
    density_field,
    bsdf=learned_bsdf,
    integrator=integrator,
    lights=lights,
    Rs=Rs,
    Ts=Ts,
    exp_imgs=exp_imgs,
    exp_masks=exp_masks,
    opt=opt,
    size=SIZE,
    crop_size=128,
    save_freq=7500,
    valid_freq=4000,
    max_valid_size=128,
    iters=iters,
    N=4,
    extra_loss=extra_loss,
    uv_select=lambda _, crop_size: rand_uv(SIZE, SIZE, crop_size),
    light_update=light_update,

    name_fn=lambda i: f"outputs/train_{k}_{i:06}.png",
    valid_name_fn=lambda i: f"outputs/valid_{k}_{i:06}.png",

    silent=True,
    really_silent=True,
    w_isect=occ_mlp,
  )

  if iters > 0:
    torch.jit.save(density_field.sdf, f"models/col_{k}_sdf.pt")
    torch.save(learned_bsdf, f"models/col_{k}_bsdf.pt")

  print("Checking train set")

  # Training set
  test(
    density_field,
    integrator=integrator,
    bsdf=learned_bsdf,
    lights=lights,
    Rs=Rs,
    Ts=Ts,
    exp_imgs=exp_imgs,
    size=SIZE,
    light_update=light_update,
    name_fn=lambda i: f"outputs/col_final_{k}_{i:03}.png",

    w_isect=True,
  )

  Rs, Ts, exp_imgs, exp_masks, xyzs = test_colocate_resources(k, SIZE, dist=DIST, device=device)

  xyzs_iter = iter(xyzs)
  def light_update(_, light): light.location = next(xyzs_iter).unsqueeze(0)

  print("Starting test set")

  # Test set
  test(
    density_field,
    integrator=integrator,
    bsdf=learned_bsdf,
    lights=lights,
    Rs=Rs, Ts=Ts,
    exp_imgs=exp_imgs,
    size=SIZE,
    light_update=light_update,

    name_fn=lambda i: f"outputs/col_test_{k}_{i:03}.png",
  )
kinds = ["buddha", "teapot", "bunny"]
for k in kinds: train_on_kind(k)
