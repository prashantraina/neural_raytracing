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
  NeuralApprox, Direct,
)
from pytorch3d.pathtracer.shapes.sdfs import (
  SDF, SphereSDF, CapsuleSDF, RoundBoxSDF,
)
from pytorch3d.pathtracer.utils import ( mse2psnr )
from pytorch3d.pathtracer.training_utils import ( save_image, save_plot )
from pytorch3d.pathtracer.neural_blocks import ( SkipConnMLP )
from pytorch3d.pathtracer.cameras import ( NeRFCamera )
from pytorch3d.pathtracer.lights import ( PointLights )
import imageio
import json
from tqdm import tqdm
from pytorch_msssim import ( ssim, ms_ssim )

SIZE=200
dataset = "armadillo"
assert(dataset in ["armadillo", "hotdogs"])
DIR = f"nerv_public_release/{dataset}/"
var = "_sigmoid"
assert(var in ["", "_clamp", "_sigmoid"])
print(dataset, var)
with_norm = var != "_sigmoid"

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def load_image(path): return torch.from_numpy(imageio.imread(path))

integrator=Direct(training=False)

shape = torch.jit.load(f"models/nerv_{dataset}{var}_sdf.pt")
density_field = SDF(sdf=shape)
density_field.max_steps = 64
density_field.dist = 2.2 if with_norm else 8

learned_bsdf =\
  torch.load(f"models/nerv_{dataset}{var}_bsdf.pt")

#learned_bsdf = torch.load(f"nerv_hotdogs_bsdf_init.pt")
#for bsdf in learned_bsdf.bsdfs: setattr(bsdf, "act", torch.sigmoid)
#for bsdf in learned_bsdf.bsdfs: setattr(bsdf, "act", nn.Softplus())

occ_mlp = torch.load(f"models/nerv_{dataset}{var}_occ.pt")

tfs = json.load(open(DIR + "transforms_test.json"))
exp_imgs = []
exp_masks = []
light_locs = []
focal = 0.5 * SIZE / np.tan(0.5 * float(tfs['camera_angle_x']))
light_weights = [ [[1.,1.,1.]] ] * 100
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
  ll = torch.tensor([frame['light_loc']], dtype=torch.float, device=device)
  if with_norm:
    ln = torch.linalg.norm(ll, dim=-1)
    light_locs.append(ln/n * F.normalize(ll, dim=-1))
  else:
    light_locs.append(ll)

for frame in tfs["frames"][100:]:
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
    ln = torch.linalg.norm(ll, dim=-1, keepdim=True)
    light_locs.append(ln/n * F.normalize(ll, dim=-1))
  else:
    light_locs.append(ll)

  light_weights.append(frame['light_weights'])


density_field.max_steps = 256

# 50 seems to work best but worth looking at other intensities.
o_i = 100
print("other_intensity", o_i)
# test nerv point light
def test_nerv_ptl(
  density_field, bsdf,
  integrator,
  light_locs, cam_to_worlds, focal,
  light_weights,
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
      got = None
      for j, lw in enumerate(light_weights[i]):
        scale = 100 if j == 0 else o_i
        lights = PointLights(
          intensity=lw[:3],
          location=lp[j].unsqueeze(0),scale=scale,device=device
        )
        sample = pt.pathtrace(
          density_field,
          size=size, chunk_size=min(size, 100), bundle_size=1, bsdf=bsdf,
          integrator=integrator,
          # 0 is for comparison, 1 is for display
          background=0,
          cameras=cameras, lights=lights, device=device, silent=True,
          w_isect=w_isect,
        )[0].clamp(min=0, max=1)
        if got is None: got = sample
        else: got = got + sample
      got = got.clip(min=0,max=1)
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
  return

print("Occ-MLP variant:")

test_nerv_ptl(
  density_field, bsdf=learned_bsdf, integrator=integrator, light_locs=light_locs,
  exp_imgs=exp_imgs,
  cam_to_worlds=cam_to_worlds, focal=focal, size=SIZE,
  light_weights=light_weights,

  name_fn=lambda i: f"outputs/test_nerv_{dataset}{var}_occ_{i:03}.png",
  w_isect=occ_mlp,
)

# Hard shadow variant

print("Hard shadow variant:")

test_nerv_ptl(
  density_field, bsdf=learned_bsdf, integrator=integrator, light_locs=light_locs,
  exp_imgs=exp_imgs,
  cam_to_worlds=cam_to_worlds, focal=focal, size=SIZE,
  light_weights=light_weights,

  name_fn=lambda i:f"outputs/test_nerv_{dataset}{var}_hs_{i:03}.png",
  w_isect=True,
)
