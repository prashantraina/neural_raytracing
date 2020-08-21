import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
  masked_loss, LossSampler,
  rand_uv_mask, mse2psnr,
  load_image,
)
from pytorch3d.renderer import ( OpenGLPerspectiveCameras, look_at_view_transform )
from pytorch3d.pathtracer.cameras import ( NeRFCamera, DTUCamera, NeRVCamera )
from pytorch3d.pathtracer.integrators import ( NeRFIntegrator, Mask )
from pytorch3d.pathtracer.lights import ( PointLights )
from tqdm import tqdm, trange
import numpy as np
import pytorch3d.pathtracer as pt
import matplotlib.pyplot as plt
from pytorch_msssim import ( ssim, ms_ssim )
import json

def save_image(name, img): plt.imsave(name, img.detach().cpu().clamp(0,1).numpy())
def save_plot(expected, got, name):
  fig=plt.figure()
  fig.add_subplot(1, 2, 1)
  plt.imshow(got.detach().squeeze().cpu().numpy())
  plt.grid("off");
  plt.axis("off");
  fig.add_subplot(1, 2, 2)
  plt.imshow(expected.detach().squeeze().cpu().numpy())
  plt.grid("off");
  plt.axis("off");
  plt.savefig(name)
  plt.close(fig)

def pathtrace_labels(ref, size, integrator, bsdf, lights, Rs, Ts):
  exp_imgs = []
  exp_masks = []
  with torch.no_grad():
    for i, (R, T) in enumerate(zip(Rs, Ts)):
      device=R.device
      cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
      expected = pt.pathtrace(
        ref,
        size=size, chunk_size=size, bundle_size=1,
        bsdf=bsdf, integrator=Mask(integrator),
        cameras=cameras, lights=lights,
        device=device, silent=True,
      )[0].detach()
      exp_imgs.append(expected[..., :-1])
      exp_masks.append(expected[..., -1])
  return exp_imgs, exp_masks

def no_update(cameras, lights): return

def train(
  shape,
  bsdf,
  integrator,
  lights,
  Rs, Ts,
  exp_imgs,
  exp_masks,
  opt,
  size,
  N=3,
  iters=50_000,
  num_ckpts=5,
  save_freq=50,
  # light update can be used if colocated lights are used with a camera
  light_update=no_update,
  save_fn=lambda i: None,
  name_fn=lambda i: f"outputs/train_{i:05}.png",
  extra_loss=lambda mi, got, exp, mask: 0,
  silent=True,
):
  integrator = NeRFIntegrator(integrator)
  device = exp_imgs[0].device
  ckpt_freq = (iters//num_ckpts) - 1
  losses=[]
  selector = LossSampler(len(exp_imgs))

  iterator = range(iters)
  if not silent: iterator = trange(iters)
  update = lambda loss, _: iterator.set_postfix(refresh=False, loss=f"{loss:.05}")
  if silent: update = lambda loss, i: print(f"{i:06}: {loss:.05}")

  for i in iterator:
    idxs = selector.sample(n=N)
    R = torch.cat([Rs[i] for i in idxs], dim=0)
    T = torch.cat([Ts[i] for i in idxs], dim=0)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
    light_update(cameras, lights)

    opt.zero_grad()
    got, mi = pt.pathtrace(
      shape, size=size, chunk_size=size, bundle_size=1,
      bsdf=bsdf, integrator=integrator,
      cameras=cameras, lights=lights,
      device=device,
      background=0,
      addition = lambda mi: mi,
      squeeze_first=False,
      silent=True,
    )
    if (i % save_freq) == 0: save_image(name_fn(i), got[0])
    exp = torch.stack([exp_imgs[i] for i in idxs])
    mask = torch.stack([exp_masks[i] for i in idxs])
    loss = masked_loss(
      got[..., :3], exp, mi.throughput.squeeze(-1), mask, mask_weight=15,
      with_logits=mi.with_logits,
    ) + extra_loss(mi, got, exp, mask)
    if loss.isnan(): continue

    loss.backward()
    loss = loss.item()
    selector.update_idxs(idxs, loss)
    losses.append(loss)
    opt.step()
    update(loss, i)
    if ((i % ckpt_freq) == 0) and (i != 0): save_fn(i)
  return losses

def train_sample(
  shape,
  bsdf,
  integrator,
  lights,
  Rs, Ts,
  exp_imgs, exp_masks,
  opt, size, crop_size,
  N=3, iters=50_000,
  num_ckpts=5, save_freq=50,
  valid_freq=250, max_valid_size=128,
  extra_loss=lambda mi, got, exp, mask: 0,
  save_fn=lambda i: None,
  name_fn=lambda i: f"outputs/train_{i:05}.png",
  valid_name_fn=lambda i: f"outputs/valid_{i:05}.png",
  uv_select=lambda mask, crop_size: rand_uv_mask(mask, crop_size),
  light_update=no_update,
  silent=False,
  really_silent=False,
  w_isect=False,
):
  device = exp_imgs[0].device
  ckpt_freq = (iters//num_ckpts) - 1
  losses=[]
  selector = LossSampler(len(exp_imgs))

  iterator = range(iters)
  if not silent: iterator = trange(iters)
  update = lambda loss, _: iterator.set_postfix(refresh=False, loss=f"{loss:.05}")
  if silent: update = lambda loss, i: print(f"{i:06}: {loss:.05}")
  if really_silent: update=lambda loss,i:print(f"{i:06}: {loss:.05}")if i%1000==0 else None

  for i in iterator:
    idxs = selector.sample(n=N)
    R = torch.cat([Rs[i] for i in idxs], dim=0)
    T = torch.cat([Ts[i] for i in idxs], dim=0)
    exp = torch.stack([exp_imgs[i] for i in idxs])
    mask = torch.stack([exp_masks[i] for i in idxs])
    cameras = mk_camera(R,T,focal,device)
    light_update(cameras, lights)

    opt.zero_grad()
    (u, v) = uv_select(mask[0], crop_size)
    got, mi = pt.pathtrace_sample(
      shape, size=size, chunk_size=size, bundle_size=1,
      crop_size=crop_size,
      bsdf=bsdf, integrator=integrator,
      cameras=cameras, lights=lights,
      device=device,
      uv=(u,v),
      addition = lambda mi: mi,
      squeeze_first=False, silent=True,
      w_isect=w_isect,
    )
    if (i % save_freq) == 0: save_image(name_fn(i), got[0])
    exp = exp[:, u:u+crop_size,v:v+crop_size]
    mask = mask[:, u:u+crop_size, v:v+crop_size]
    loss = masked_loss(
      got[..., :3], exp, mi.throughput.squeeze(-1), mask, mask_weight=15,
      with_logits=mi.with_logits,
    ) + extra_loss(mi, got, exp, mask)
    if loss.isnan(): continue

    loss.backward()
    opt.step()
    loss = loss.detach().item()
    losses.append(loss)
    update(loss, i)

    if ((i % ckpt_freq) == 0) and (i != 0): save_fn(i)

    if (i % valid_freq) == 0:
      with torch.no_grad():
        R = R[0].unsqueeze(0)
        T = T[0].unsqueeze(0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        light_update(cameras, lights)
        validate, _ = pt.pathtrace(
          shape, size=size, chunk_size=min(size, max_valid_size),
          bundle_size=1,
          bsdf=bsdf, integrator=NeRFIntegrator(integrator),
          cameras=cameras, lights=lights, device=device,silent=True,
          w_isect=w_isect,
        )
        save_image(valid_name_fn(i), validate)
  return losses

# training specifically with the NeRF camera and setup
def train_nerf(
  shape,
  bsdf,
  integrator,
  lights,
  cam_to_worlds,
  focal,
  exp_imgs,
  exp_masks,
  opt,
  size,
  crop_size,
  N=3,
  iters=50_000,
  num_ckpts=5,
  save_freq=50,
  valid_freq=250,
  max_valid_size=128,
  extra_loss=lambda mi, got, exp, mask: 0,
  save_fn=lambda i: None,
  name_fn=lambda i: f"outputs/train_{i:05}.png",
  valid_name_fn=lambda i: f"outputs/valid_{i:05}.png",
  uv_select=lambda mask, crop_size: rand_uv_mask(mask, crop_size),
  silent=False,
):
  train_integrator = NeRFIntegrator(integrator)
  device = exp_imgs[0].device
  ckpt_freq = (iters//num_ckpts) - 1
  losses=[]
  selector = LossSampler(len(exp_imgs))

  iterator = range(iters)
  if not silent: iterator = trange(iters)
  update = lambda loss, _: iterator.set_postfix(refresh=False, loss=f"{loss:.05}")
  if silent: update = lambda loss, i: print(f"{i:06}: {loss:.05}")

  for i in iterator:
    idxs = selector.sample(n=N)
    c2w = torch.stack([cam_to_worlds[i] for i in idxs], dim=0)
    exp = torch.stack([exp_imgs[i] for i in idxs])
    mask = torch.stack([exp_masks[i] for i in idxs])
    cameras = NeRFCamera(cam_to_world=c2w, focal=focal, device=device)

    opt.zero_grad()
    (u, v) = uv_select(mask[0], crop_size)
    got, mi = pt.pathtrace_sample(
      shape, size=size, chunk_size=size, bundle_size=1,
      crop_size=crop_size,
      bsdf=bsdf, integrator=train_integrator,
      cameras=cameras, lights=lights,
      device=device,
      uv=(u,v),
      background=0,
      addition = lambda mi: mi,
      squeeze_first=False, silent=True,
    )
    if (i % save_freq) == 0: save_image(name_fn(i), got[0])
    exp = exp[:, u:u+crop_size,v:v+crop_size]
    mask = mask[:, u:u+crop_size, v:v+crop_size]
    loss = masked_loss(
      got[..., :3], exp, mi.throughput.squeeze(-1), mask, mask_weight=15,
      with_logits=mi.with_logits,
    ) + extra_loss(mi, got, exp, mask)
    if loss.isnan():
      # don't know if need to propagate NaN to get exception
      loss.backward()
      opt.step()
      raise Exception("Unexpected NaN")

    loss.backward()
    opt.step()
    loss = loss.detach().item()
    losses.append(loss)
    selector.update_idxs(idxs, loss)

    update(loss, i)
    if ((i % ckpt_freq) == 0) and (i != 0): save_fn(i)

    if (i % valid_freq) == 0:
      with torch.no_grad():
        c2w = c2w[0].unsqueeze(0)
        cameras = NeRFCamera(cam_to_world=c2w, focal=focal, device=device)
        validate, _ = pt.pathtrace(
          shape, size=size, chunk_size=min(size, max_valid_size),
          bundle_size=1,
          bsdf=bsdf, integrator=train_integrator,
          cameras=cameras, lights=lights, device=device,silent=True,
        )
        save_image(valid_name_fn(i), validate)
  return losses

def test_nerf(
  density_field,
  integrator,
  bsdf,
  lights,
  cam_to_worlds,
  focal,
  exp_imgs,
  size,

  name_fn = lambda i: f"outputs/test_{i:03}.png"
):
  device=exp_imgs[0].device
  l1_losses = []
  l2_losses = []
  psnr_losses = []
  gots = []
  with torch.no_grad():
    for i, c2w in enumerate(tqdm(cam_to_worlds)):
      exp = exp_imgs[i]
      cameras = NeRFCamera(cam_to_world=c2w.unsqueeze(0), focal=focal, device=device)
      got = pt.pathtrace(
        density_field,
        size=size, chunk_size=min(size, 256), bundle_size=1, bsdf=bsdf,
        integrator=integrator,
        cameras=cameras, lights=lights, device=device, silent=True,
        background=0,
      )[0].clamp(min=0, max=1)
      save_plot(exp, got, name_fn(i))
      l1_losses.append(F.l1_loss(exp,got).item())
      l2_losses.append(F.mse_loss(exp,got).item())
      psnr_losses.append(mse2psnr(F.mse_loss(exp, got)).item())
      gots.append(got)
  print("Avg l1 loss", np.mean(l1_losses))
  print("Avg l2 loss", np.mean(l2_losses))
  print("Avg PSNR loss", np.mean(psnr_losses))
  with torch.no_grad():
    gots = torch.stack(gots, dim=0).permute(0, 3, 1, 2)
    exps = torch.stack(exp_imgs, dim=0).permute(0, 3, 1, 2)
    torch.cuda.empty_cache()
    ssim_loss = ssim(gots, exps, data_range=1, size_average=True).item()
    print("SSIM loss", ssim_loss)
  return

# training specifically with DTU camera and setup
def train_dtu(
  shape,
  bsdf,
  integrator,
  lights,
  poses, intrinsics,
  exp_imgs,
  exp_masks,
  opt,
  size, crop_size,
  N=3,
  iters=50_000,
  num_ckpts=5, save_freq=50,
  valid_freq=250, max_valid_size=128,
  extra_loss=lambda mi, got, exp, mask: 0,
  save_fn=lambda i: None,
  name_fn=lambda i: f"outputs/train_{i:05}.png",
  valid_name_fn=lambda i: f"outputs/valid_{i:05}.png",
  uv_select=lambda mask, crop_size: rand_uv_mask(mask, crop_size),
  silent=False,
):
  train_integrator = NeRFIntegrator(integrator)
  device = exp_imgs[0].device
  ckpt_freq = (iters//num_ckpts) - 1
  losses=[]
  selector = LossSampler(len(exp_imgs))

  iterator = range(iters)
  if not silent: iterator = trange(iters)
  update = lambda loss, _: iterator.set_postfix(refresh=False, loss=f"{loss:.05}")
  if silent: update = lambda loss, i: print(f"{i:06}: {loss:.05}")

  for i in iterator:
    idxs = selector.sample(n=N)
    pose = torch.stack([poses[i] for i in idxs], dim=0)
    intrinsic = torch.stack([intrinsics[i] for i in idxs], dim=0)
    exp = torch.stack([exp_imgs[i] for i in idxs])
    mask = torch.stack([exp_masks[i] for i in idxs])
    cameras = DTUCamera(pose=pose, intrinsic=intrinsic, device=device)

    opt.zero_grad()
    (u, v) = uv_select(mask[0], crop_size)
    got, mi = pt.pathtrace_sample(
      shape, size=size, chunk_size=size, bundle_size=1,
      crop_size=crop_size,
      bsdf=bsdf, integrator=train_integrator,
      cameras=cameras, lights=lights,
      device=device,
      uv=(u,v),
      background=0,
      addition = lambda mi: mi,
      squeeze_first=False, silent=True,
    )
    if (i % save_freq) == 0: save_image(name_fn(i), got[0])
    exp = exp[:, u:u+crop_size,v:v+crop_size]
    mask = mask[:, u:u+crop_size, v:v+crop_size]
    loss = masked_loss(
      got[..., :3], exp, mi.throughput.squeeze(-1), mask, mask_weight=10,
      with_logits=mi.with_logits,
    ) + extra_loss(mi, got, exp, mask)
    if loss.isnan():
      # don't know if need to propagate NaN to get exception
      loss.backward()
      opt.step()
      raise Exception("Unexpected NaN")

    loss.backward()
    opt.step()
    loss = loss.detach().item()
    losses.append(loss)
    selector.update_idxs(idxs, loss)
    update(loss, i)

    if ((i % ckpt_freq) == 0) and (i != 0): save_fn(i)

    if (i % valid_freq) == 0:
      with torch.no_grad():
        pose = pose[None, 0]
        intrinsic = intrinsics[None, 0]
        cameras = DTUCamera(pose=pose, intrinsic=intrinsic, device=device)
        validate, _ = pt.pathtrace(
          shape, size=size, chunk_size=min(size, max_valid_size),
          bundle_size=1,
          bsdf=bsdf, integrator=train_integrator,
          cameras=cameras, lights=lights, device=device,silent=True,
        )
        save_image(valid_name_fn(i), validate)
  return losses

def test_dtu(
  density_field,
  integrator,
  bsdf,
  lights,
  poses, intrinsics,
  exp_imgs,
  exp_masks,
  size,

  name_fn = lambda i: f"outputs/test_{i:03}.png"
):
  device=exp_imgs[0].device
  l1_losses = []
  l2_losses = []
  psnr_losses = []
  gots = []
  exps = []
  with torch.no_grad():
    for i, (pose, intrinsic) in enumerate(zip(tqdm(poses), intrinsics)):
      exp = exp_imgs[i]; mask = (exp_masks[i] == 1)
      cameras = DTUCamera(pose=pose[None, ...], intrinsic=intrinsic[None, ...], device=device)
      got = pt.pathtrace(
        density_field,
        size=size, chunk_size=min(size, 128), bundle_size=1, bsdf=bsdf,
        integrator=integrator,
        cameras=cameras, lights=lights, device=device, silent=True,
        background=0,
      )[0].clamp(min=0, max=1)
      # IDR also scales by 255 here so check if that affects the output
      save_plot(exp, got, name_fn(i))
      exp = exp * mask[..., None]
      got = got * mask[..., None]
      l1_losses.append(F.l1_loss(exp, got).item())
      mse = F.mse_loss(exp, got)
      l2_losses.append(mse.item())
      psnr_losses.append(mse2psnr(mse).item())
      gots.append(got)
      exps.append(exp)
  print("Avg l1 loss", np.mean(l1_losses))
  print("Avg l2 loss", np.mean(l2_losses))
  print("Avg PSNR loss", np.mean(psnr_losses))
  with torch.no_grad():
    # takes a lot of memory
    gots = torch.stack(gots, dim=0).permute(0, 3, 1, 2)
    exps = torch.stack(exps, dim=0).permute(0, 3, 1, 2)
    torch.cuda.empty_cache()
    ssim_loss = ssim(gots, exps, data_range=1, size_average=True).item()
    print("SSIM loss", ssim_loss)
  return

def test(
  density_field,
  integrator,
  bsdf,
  lights,
  Rs, Ts,
  exp_imgs,
  size,
  max_chunk_size=128,
  light_update=no_update,
  name_fn=lambda i: f"outputs/test_{i:03}.png",
  w_isect=False,
):
  device=exp_imgs[0].device
  l1_losses = []
  l2_losses = []
  psnr_losses = []
  ssim_losses = []
  gots = []
  with torch.no_grad():
    for i, (R, T) in enumerate(zip(tqdm(Rs), Ts)):
      exp = exp_imgs[i]
      cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
      light_update(cameras, lights)
      got = pt.pathtrace(
        density_field,
        size=size, chunk_size=min(size, max_chunk_size), bundle_size=1, bsdf=bsdf,
        integrator=integrator,
        cameras=cameras, lights=lights, device=device, silent=True,
        background=0,
        w_isect=w_isect,
      )[0].clamp(min=0, max=1)
      save_plot(exp, got, name_fn(i))
      l1_losses.append(F.l1_loss(exp, got).item())
      l2_losses.append(F.mse_loss(exp, got).item())
      psnr_losses.append(mse2psnr(F.mse_loss(exp, got)).item())
      gots.append(got)
  print("Avg l1 loss", np.mean(l1_losses))
  print("Avg l2 loss", np.mean(l2_losses))
  print("Avg PSNR loss", np.mean(psnr_losses))
  with torch.no_grad():
    # takes a lot of memory
    gots = torch.stack(gots[::3], dim=0).permute(0, 3, 1, 2)
    exps = torch.stack(exp_imgs[::3], dim=0).permute(0, 3, 1, 2)
    torch.cuda.empty_cache()
    ssim_loss = ssim(gots, exps, data_range=1, size_average=True).item()
    print("SSIM loss", ssim_loss)
  return


# common utility for getting required items for testing colocation
def test_colocate_resources(
  kind,
  size=128,
  dist=1,
  device="cuda",
):
  def elaz_to_xyz(elev, azim, rad):
    elev = torch.deg2rad(elev)
    azim = torch.deg2rad(azim)
    x = rad * elev.cos() * azim.sin()
    y = rad * elev.cos() * azim.cos()
    z = rad * elev.sin()
    return torch.stack([x,y,z], dim=0)

  Rs = []; Ts = []
  exp_imgs = []
  exp_masks = []
  xyzs = []
  for i, elev in enumerate(torch.linspace(0, 45, 4, device=device)):
    for j, azim in enumerate(torch.linspace(-90, 90, 4, device=device)):
      R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
      for k, e2 in enumerate(torch.linspace(0, 45, 3, device=device)):
        for l, a2 in enumerate(torch.linspace(-90, 90, 3, device=device)):
          Rs.append(R); Ts.append(T)
          img = load_image(
            f"mitsuba_scenes/cbox_relight/gt_{kind}_{i:03}_{j:03}_{k:03}_{l:03}.png",
            (size, size)
          ).to(device)
          exp_imgs.append(img[..., :3])
          exp_masks.append(img[..., 3])
          xyzs.append(elaz_to_xyz(e2, a2, dist * 1.05))

  return Rs, Ts, exp_imgs, exp_masks, xyzs

def test_nerf_resources(
  directory,
  size=128,
  kind = "test",
  device="cuda"
):
  assert(kind in ["train", "test"])
  tfs = json.load(open(directory + f"transforms_{kind}.json"))
  exp_imgs = []
  exp_masks = []
  focal = 0.5 * size / np.tan(0.5 * float(tfs['camera_angle_x']))
  cam_to_worlds=[]
  for frame in tfs["frames"]:
    img = load_image(os.path.join(directory,frame['file_path']+'.png'),resize=(size,size))\
      .to(device)
    exp_imgs.append(img[..., :3])
    exp_masks.append((img[..., 3] - 1e-5).ceil())
    tf_mat = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)[:3, :4]
    # set distance to 1 from origin
    tf_mat[:3, 3] = F.normalize(tf_mat[:3, 3], dim=-1)
    cam_to_worlds.append(tf_mat)

  return cam_to_worlds, focal, exp_imgs, exp_masks

# Specific layout for testing on the NeRV dataset
def train_nerv(
  shape, bsdf,
  integrator,
  lights,
  world_to_cams, locs, focal,
  exp_imgs, exp_masks,
  opt,
  size, crop_size, N=3,
  iters=50_000,
  num_ckpts=5,
  save_freq=50, valid_freq=250,
  max_valid_size=128,
  extra_loss=lambda mi, got, exp, mask: 0,
  save_fn=lambda i: None,
  name_fn=lambda i: f"outputs/train_{i:05}.png",
  valid_name_fn=lambda i: f"outputs/valid_{i:05}.png",
  uv_select=lambda mask, crop_size: rand_uv_mask(mask, crop_size),
  silent=False,
):
  train_integrator = NeRFIntegrator(integrator)
  device = exp_imgs[0].device
  ckpt_freq = (iters//num_ckpts) - 1
  losses=[]
  selector = LossSampler(len(exp_imgs))

  iterator = range(iters)
  if not silent: iterator = trange(iters)
  update = lambda loss, _: iterator.set_postfix(refresh=False, loss=f"{loss:.05}")
  if silent: update = lambda loss, i: print(f"{i:06}: {loss:.05}")

  for i in iterator:
    idxs = selector.sample(n=N)
    w2c = torch.stack([world_to_cams[i] for i in idxs], dim=0)
    loc = torch.stack([locs[i] for i in idxs], dim=0)

    exp = torch.stack([exp_imgs[i] for i in idxs])
    mask = torch.stack([exp_masks[i] for i in idxs])
    cameras = NeRVCamera(world_to_cam=w2c, loc=loc, focal=focal, device=device)

    opt.zero_grad()
    (u, v) = uv_select(mask[0], crop_size)
    got, mi = pt.pathtrace_sample(
      shape, size=size, chunk_size=size, bundle_size=1,
      crop_size=crop_size,
      bsdf=bsdf, integrator=train_integrator,
      cameras=cameras, lights=lights,
      device=device,
      uv=(u,v),
      background=0,
      addition = lambda mi: mi,
      squeeze_first=False, silent=True,
    )
    if (i % save_freq) == 0: save_image(name_fn(i), got[0])
    exp = exp[:, u:u+crop_size,v:v+crop_size]
    mask = mask[:, u:u+crop_size, v:v+crop_size]
    loss = masked_loss(
      got[..., :3], exp, mi.throughput.squeeze(-1), mask, mask_weight=10,
      with_logits=mi.with_logits,
    ) + extra_loss(mi, got, exp, mask)
    if loss.isnan():
      # don't know if need to propagate NaN to get exception
      loss.backward()
      opt.step()
      raise Exception("Unexpected NaN")

    loss.backward()
    opt.step()
    loss = loss.detach().item()
    losses.append(loss)
    selector.update_idxs(idxs, loss)

    update(loss, i)
    if ((i % ckpt_freq) == 0) and (i != 0): save_fn(i)

    if (i % valid_freq) == 0:
      with torch.no_grad():
        w2c = w2c[0].unsqueeze(0)
        loc = loc[0].unsqueeze(0)
        cameras = NeRVCamera(world_to_cam=w2c, loc=loc, focal=focal, device=device)
        validate, _ = pt.pathtrace(
          shape, size=size, chunk_size=min(size, max_valid_size),
          bundle_size=1,
          bsdf=bsdf, integrator=train_integrator,
          cameras=cameras, lights=lights, device=device,silent=True,
        )
        save_image(valid_name_fn(i), validate)
  return losses

# training specifically with the NeRF camera and setup
def train_nerv_ptl(
  shape,
  bsdf,
  integrator,
  cam_to_worlds,
  light_locs,
  focal,
  exp_imgs, exp_masks,
  opt,
  size, crop_size,
  N=3,
  iters=50_000,
  num_ckpts=3,
  save_freq=10_000,
  valid_freq=250,
  max_valid_size=128,
  extra_loss=lambda mi, got, exp, mask: 0,
  save_fn=lambda i: None,
  name_fn=lambda i: f"outputs/train_{i:05}.png",
  valid_name_fn=lambda i: f"outputs/valid_{i:05}.png",
  uv_select=lambda mask, crop_size: rand_uv_mask(mask, crop_size),
  silent=False,
  w_isect=True,
):
  train_integrator = NeRFIntegrator(integrator)
  device = exp_imgs[0].device
  ckpt_freq = (iters//num_ckpts) - 1
  losses=[]
  selector = LossSampler(len(exp_imgs))

  iterator = range(iters)
  if not silent: iterator = trange(iters)
  update = lambda loss, _: iterator.set_postfix(refresh=False, loss=f"{loss:.05}")
  if silent: update = lambda loss, i: print(f"{i:06}: {loss:.05}") if i % 10 == 0 else None

  #use_repeat = False
  #repeats = 0
  for i in iterator:
    #if not use_repeat:
    idxs = selector.sample(n=N)
    c2w = torch.stack([cam_to_worlds[i] for i in idxs], dim=0)
    exp = torch.stack([exp_imgs[i] for i in idxs])
    mask = torch.stack([exp_masks[i] for i in idxs])
    light_pos = torch.stack([light_locs[i] for i in idxs])
    cameras = NeRFCamera(cam_to_world=c2w, focal=focal, device=device)
    lights = PointLights(intensity=[1,1,1], location=light_pos, scale=100, device=device)

    opt.zero_grad()
    #if not use_repeat:
    (u, v) = uv_select(mask[0], crop_size)
    got, mi = pt.pathtrace_sample(
      shape, size=size, chunk_size=size, bundle_size=1,
      crop_size=crop_size,
      bsdf=bsdf, integrator=train_integrator,
      cameras=cameras, lights=lights,
      device=device,
      uv=(u,v),
      background=0,
      addition = lambda mi: mi,
      squeeze_first=False, silent=True,
      w_isect=w_isect,
    )
    if (i % save_freq) == 0: save_image(name_fn(i), got[0])
    exp = exp[:, u:u+crop_size,v:v+crop_size]
    mask = mask[:, u:u+crop_size, v:v+crop_size]
    loss = masked_loss(
      got[..., :3], exp, mi.throughput.squeeze(-1), mask, mask_weight=10,
      with_logits=mi.with_logits,
      tone_mapping=True,
    ) + extra_loss(mi, got, exp, mask)
    if loss.isnan():
      # don't know if need to propagate NaN to get exception
      loss.backward()
      opt.step()
      raise Exception("Unexpected NaN")

    loss.backward()
    opt.step()
    loss = loss.detach().item()
    losses.append(loss)
    selector.update_idxs(idxs, loss)

    update(loss, i)

    #use_repeat = (loss > 4) and (repeats < 100)
    #repeats += 1
    #if not use_repeat: repeats = 0

    if ((i % ckpt_freq) == 0) and (i != 0): save_fn(i)

    if (i % valid_freq) == 0:
      with torch.no_grad():
        c2w = c2w[0].unsqueeze(0)
        cameras = NeRFCamera(cam_to_world=c2w, focal=focal, device=device)
        lights.location = lights.location[0].unsqueeze(0)
        validate, _ = pt.pathtrace(
          shape, size=size, chunk_size=min(size, max_valid_size),
          bundle_size=1,
          bsdf=bsdf, integrator=train_integrator,
          cameras=cameras, lights=lights, device=device,silent=True,
          w_isect=w_isect,
        )
        save_image(valid_name_fn(i), validate ** (1/2.2))
  return losses

# test nerv point light
def test_nerv_ptl(
  density_field, bsdf,
  integrator,
  light_locs, cam_to_worlds, focal,
  exp_imgs,
  size,

  name_fn = lambda i: f"outputs/test_{i:03}.png",
  w_isect=True,
):
  device=exp_imgs[0].device
  l1_losses = []
  l2_losses = []
  psnr_losses = []
  gots = []
  with torch.no_grad():
    for i, (c2w, lp) in enumerate(zip(tqdm(cam_to_worlds), light_locs)):
      exp = exp_imgs[i].clamp(min=0, max=1)
      cameras = NeRFCamera(cam_to_world=c2w.unsqueeze(0), focal=focal, device=device)
      lights = PointLights(intensity=[1,1,1], location=lp[None,...], scale=100, device=device)
      got = pt.pathtrace(
        density_field,
        size=size, chunk_size=min(size, 100), bundle_size=1, bsdf=bsdf,
        integrator=integrator,
        # 0 is for comparison, 1 is for display
        background=0,
        cameras=cameras, lights=lights, device=device, silent=True,
        w_isect=w_isect,
      )[0].clamp(min=0, max=1)
      save_plot(
        exp ** (1/2.2), got ** (1/2.2), name_fn(i)
      )
      l1_losses.append(F.l1_loss(exp,got).item())
      mse = F.mse_loss(exp,got)
      l2_losses.append(mse.item())
      psnr = mse2psnr(mse).item()
      psnr_losses.append(psnr)
      gots.append(got)
  print("Avg l1 loss", np.mean(l1_losses))
  print("Avg l2 loss", np.mean(l2_losses))
  print("Avg PSNR loss", np.mean(psnr_losses))
  with torch.no_grad():
    # takes a lot of memory
    gots = torch.stack(gots, dim=0).permute(0, 3, 1, 2)
    tm_gots = gots/(1+gots)
    exps = torch.stack(exp_imgs, dim=0).permute(0, 3, 1, 2)
    tm_exps = exps/(1+exps)
    torch.cuda.empty_cache()
    ssim_loss = ms_ssim(tm_gots, tm_exps, data_range=1, size_average=True).item()
    print("MS-SSIM loss", ssim_loss)

    ssim_loss = ssim(tm_gots, tm_exps, data_range=1, size_average=True).item()
    print("SSIM loss", ssim_loss)
    #max_v = tm_exps.max().item()
    #ssim_loss = ms_ssim(
    #  tm_gots.clamp(min=0, max=max_v),
    #  tm_exps,
    #  data_range=max_v,
    #  size_average=True
    #).item()
    #print("MS-SSIM with lower-range loss", ssim_loss)
  return
