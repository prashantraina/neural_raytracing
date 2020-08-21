import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import random
import pytorch3d.pathtracer as pt
import itertools

from pytorch3d.pathtracer.neural_blocks import Discriminator
from pytorch3d.pathtracer.main import pathtrace
from pytorch3d.pathtracer.utils import spherical_positions
from tqdm import trange
from pytorch3d.renderer import OpenGLPerspectiveCameras
from pytorch3d.pathtracer.shapes.nerf import ( PlainNeRF, PartialNeRF )
from pytorch3d.pathtracer.integrators import NeRFReproduce
from pytorch3d.pathtracer.training_utils import save_image


device = "cpu"
if torch.cuda.is_available(): device = "cuda:0"
device = torch.device(device)

latent_size = 10


disc = Discriminator().to(device)

#nerf_hierarchy = [
#  PartialNeRF(device=device).to(device),
#  PartialNeRF(device=device).to(device),
#  PartialNeRF(device=device).to(device),
#]

def render(camera, latent, upto=1):
  rgb, alpha = None, None
  curr_size = 8
  for i in range(min(upto, len(nerf_hierarchy))):
    rays = cameras.samples_positions(
      sampler=None, bundle_size=1, size=curr_size, N=len(cameras),
      with_noise=0.1,
    )
    nerf = nerf_hierarchy[i]
    nerf.assign_latent(latent)
    rgb_o, alpha_o = nerf(rays, 16)

    # upsample previous and combine with new
    if rgb is None:
      rgb = rgb_o
      alpha = alpha_o
    else:
      rgb = rgb_o + F.interpolate(alpha, out.shape)
      alpha = alpha_o + F.interpolate(alpha, out.shape)
    curr_size *= 2
  img = PartialNeRF.volumetric_integrate(rgb, alpha)
  return img

def parameters():
  return itertools.chain(*[nerf.parameters() for nerf in nerf_hierarchy])

pnerf = PlainNeRF(latent_size=latent_size, device=device).to(device)

Rs, Ts = spherical_positions(
  min_elev=0, max_elev = 1, min_azim=-1, max_azim=1,
  n_elev=3, n_azim=3,
  device=device,
)

#img = torch.rand(5, 3, 64, 64, device=device)
#out = disc(img)
#print(out.shape)

def train_gan(
  nerf,
  nerf_optim,
  disc,
  disc_optim,

  dataloader,

  batch_size = 3,
  iters = 80,
  device="cuda",
  valid_freq=250,
):
  integrator = NeRFReproduce()
  with trange(iters * len(dataloader)) as t:
    for j in range(iters):
      for i, (data, _tgt) in enumerate(dataloader):
        if data.shape[0] != batch_size:
          t.update()
          continue
        data = data.to(device)

        # train discriminator
        # real data:
        disc.zero_grad()

        pred = disc(data)
        label = torch.ones(batch_size, device=device)
        real_loss = F.binary_cross_entropy_with_logits(pred, label)
        real_loss.backward()
        real_loss = real_loss.item()
        # fake data:
        nerf.assign_latent(torch.randn(batch_size, latent_size, device=device))
        v = random.sample(range(Rs.shape[0]), batch_size)
        R, T = Rs[v], Ts[v]
        cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)
        fake = pt.pathtrace(
          nerf,
          size=64,
          chunk_size=8,
          bundle_size=1,
          integrator=integrator,
          cameras=cameras,
          background=1,
          bsdf=None, lights=None,
          silent=True,
          with_noise=False,
          device=device
        )[0].permute(0,3,1,2)

        pred = disc(fake.detach().clone())
        label = torch.zeros(batch_size, device=device)
        fake_loss = F.binary_cross_entropy_with_logits(pred, label)
        fake_loss.backward()
        fake_loss = fake_loss.item()

        disc_optim.step()

        # train generator/nerf
        nerf.zero_grad()
        pred = disc(fake)
        gen_loss = F.binary_cross_entropy_with_logits(pred, torch.ones_like(label))
        gen_loss = gen_loss
        gen_loss.backward()
        gen_loss = gen_loss.item()
        nerf_optim.step()

        t.set_postfix(Dreal=f"{real_loss:.05}", Dfake=f"{fake_loss:.05}", G=f"{gen_loss:.05}")
        t.update()

        ij = i + j*len(dataloader)
        if ij % valid_freq == 0:
          save_image(f"outputs/gan_valid_{ij:05}.png", fake[0].permute(1,2,0))
          #save_image(f"outputs/ref_{ij:05}.png", data[0].permute(1,2,0))
        ...
      ...
    ...
  ...

nerf_optim = optim.AdamW(pnerf.parameters(), lr=8e-5, weight_decay=0)
disc_optim = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.99))
batch_size = 5
#test_data = torch.rand(10, batch_size, 3, 64, 64,  device=device)
#dataset = torchvision.datasets.STL10(
#  'stl10',
#  download=True,
#  transform=torchvision.transforms.Compose([
#    torchvision.transforms.ToTensor(),
#    torchvision.transforms.Resize([64,64]),
#  ]),
#)
dataset = torchvision.datasets.ImageFolder(
  'flowers',
  transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize([64,64]),
  ]),
)

#dataset = torchvision.datasets.CelebA(
#  'celeba',
#  #download=True,
#  transform=torchvision.transforms.Compose([
#    torchvision.transforms.ToTensor(),
#    torchvision.transforms.Resize([64,64]),
#  ]),
#)
# TODO add transform to make small
dataloader = torch.utils.data.DataLoader(
  dataset,
  batch_size=batch_size,
)
pnerf.assign_latent(torch.randn(batch_size, latent_size, device=device))
train_gan(
  pnerf,
  nerf_optim,
  disc,
  disc_optim,

  dataloader,
  batch_size=batch_size,
)
