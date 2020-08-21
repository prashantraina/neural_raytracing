import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import pytorch3d.pathtracer as pt
from pytorch3d.pathtracer.integrators import ( NeRFReproduce )
from pytorch3d.pathtracer.shapes.sdfs import ( SDF, SphereSDF )
from pytorch3d.pathtracer.training_utils import (
  save_image, test, test_colocate_resources,
)
from pytorch3d.pathtracer.shapes.nerf import ( NeRFLE )
from pytorch3d.pathtracer.utils import (
  LossSampler, masked_loss, rand_uv, depth_image, eikonal_loss,
  load_image,
)
from pytorch3d.pathtracer.lights import ( LightField, PointLights )

# Data structures and functions for rendering
from pytorch3d.renderer import (
  look_at_view_transform, OpenGLPerspectiveCameras, HardPhongShader, look_at_rotation,
)
from tqdm import trange, tqdm

SIZE=256
N_VIEWS=8
DIST=1

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def train_on_kind(k, envmap=True):
  Rs = []; Ts = []
  exp_imgs = []
  exp_masks = []
  for i, elev in enumerate(torch.linspace(0, 45, N_VIEWS, device=device)):
    for j, azim in enumerate(torch.linspace(-90, 90, N_VIEWS, device=device)):
      R, T = look_at_view_transform(dist=DIST, elev=elev, azim=azim, device=device)
      Rs.append(R); Ts.append(T)
      img = load_image(f"mitsuba_scenes/cbox_relight/{k}_{i:03}_{j:03}.png", (SIZE, SIZE)).to(device)
      exp_imgs.append(img[..., :3])

  nerfle = NeRFLE(envmap=envmap, device=device)

  integrator = NeRFReproduce()

  lights = PointLights(device=device, scale=10)

  surface_lr = 8e-5
  print(f"NeRF({k}, envmap={envmap}) LR is {surface_lr}")
  opt = torch.optim.AdamW([
    { 'params': nerfle.parameters(), 'lr':surface_lr, },
  ], lr=surface_lr, weight_decay=0)

  def light_update(cam, light):
    light.location = cam.get_camera_center()* 1.05

  def train_sample(
    shape,
    integrator,
    lights,
    Rs, Ts,
    exp_imgs,
    opt, size, crop_size,
    light_update,
    N=3, iters=40_000,
    num_ckpts=5, save_freq=50,
    valid_freq=250, max_valid_size=128,
    extra_loss=lambda mi, got, exp, mask: 0,
    save_fn=lambda i: None,
    name_fn=lambda i: f"outputs/train_{i:05}.png",
    valid_name_fn=lambda i: f"outputs/valid_{i:05}.png",
    uv_select=lambda crop_size: None,
    silent=False,
    bsdf=None,
  ):
    device = exp_imgs[0].device
    ckpt_freq = (iters//num_ckpts) - 1
    losses=[]
    selector = LossSampler(len(exp_imgs))

    iterator = range(iters)
    if not silent: iterator = trange(iters)
    update = lambda loss, _: iterator.set_postfix(refresh=False, loss=f"{loss:.05}")
    if silent: update = lambda loss, i: None

    for i in iterator:
      idxs = selector.sample(n=N)
      R = torch.cat([Rs[i] for i in idxs], dim=0)
      T = torch.cat([Ts[i] for i in idxs], dim=0)
      exp = torch.stack([exp_imgs[i] for i in idxs])
      cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
      light_update(cameras, lights)

      opt.zero_grad()
      (u, v) = uv_select(crop_size)
      got, mi = pt.pathtrace_sample(
        shape, size=size, chunk_size=size, bundle_size=1,
        crop_size=crop_size,
        bsdf=bsdf, integrator=integrator,
        cameras=cameras, lights=lights,
        device=device,
        uv=(u,v),
        addition = lambda mi: mi,
        squeeze_first=False, silent=True,
      )
      #if (i % save_freq) == 0: save_image(name_fn(i), got[0])
      exp = exp[:, u:u+crop_size,v:v+crop_size]
      loss = F.mse_loss(got, exp)
      assert(not loss.isnan())

      loss.backward()
      opt.step()
      loss = loss.detach().item()
      losses.append(loss)
      update(loss, i)

      #if ((i % ckpt_freq) == 0) and (i != 0): save_fn(i)

      if (i % valid_freq) == 0:
        with torch.no_grad():
          R = R[0].unsqueeze(0)
          T = T[0].unsqueeze(0)
          cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
          light_update(cameras, lights)
          validate, _ = pt.pathtrace(
            shape, size=size, chunk_size=min(size, max_valid_size),
            bundle_size=1,
            bsdf=bsdf, integrator=integrator,
            cameras=cameras, lights=lights, device=device, silent=True,
          )
          save_image(valid_name_fn(i), validate)
    return losses

  losses = train_sample(
    nerfle,
    integrator=integrator,
    lights=lights,
    Rs=Rs, Ts=Ts,
    exp_imgs=exp_imgs,
    opt=opt,
    size=SIZE,
    crop_size=16,
    save_freq=20000,
    valid_freq=10000,
    max_valid_size=64,
    iters=300_000,
    N=4,
    uv_select=lambda crop_size: rand_uv(SIZE, SIZE, crop_size),
    light_update=light_update,
    valid_name_fn=lambda i: f"outputs/valid_{k}_{i:06}.png",

    silent=True,
  )

  if envmap: torch.save(nerfle, f"models/nerfle_envmap_{k}.pt")
  else: torch.save(nerfle, f"models/nerfle_{k}.pt")

  test(
    nerfle,
    integrator=integrator,
    bsdf=None,
    lights=lights,
    Rs=Rs,
    Ts=Ts,
    exp_imgs=exp_imgs,
    size=SIZE,
    light_update=light_update,
    max_chunk_size=64,
    name_fn=lambda i: f"outputs/final_{k}_{i:03}.png",
  )

  Rs, Ts, exp_imgs, exp_masks, xyzs = test_colocate_resources(k, SIZE, dist=DIST, device=device)

  xyzs_iter = iter(xyzs)
  def light_update(_, light): light.location = next(xyzs_iter).unsqueeze(0)

  print("Starting test set")

  test(
    nerfle,
    integrator=integrator,
    bsdf=None,
    lights=lights,
    Rs=Rs, Ts=Ts,
    exp_imgs=exp_imgs,
    size=SIZE,
    max_chunk_size=64,
    light_update=light_update,

    name_fn=lambda i: f"outputs/test_{k}_{envmap}_{i:03}.png",
  )

kinds = ["bunny", "teapot", "buddha"]
for k in kinds:
  train_on_kind(k, envmap=True)
  train_on_kind(k, envmap=False)
