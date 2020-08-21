import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from pytorch3d.pathtracer.shapes import Sphere
from pytorch3d.pathtracer.warps import (
  square_to_uniform_sphere,
  square_to_uniform_sphere_pdf,
  square_to_cos_hemisphere,
)
from ..interaction import ( DirectionSample, coordinate_system, from_local )
from ..neural_blocks import ( SkipConnMLP, )
from ..utils import ( nonzero_eps, )
import torch.nn.functional as F
from itertools import chain


# converts theta and phi to a direction in XYZ
def spherical_dir(theta, phi):
  s_t = theta.sin()
  c_t = (1-s_t*s_t).sqrt()
  s_p = phi.sin()
  c_p = (1-s_p*s_p).sqrt()
  return torch.stack([
    c_p * s_t,
    s_p * s_t,
    c_t,
  ], dim=-1)

# general light interface
@dataclass
class Light(nn.Module):
  def __init__(self):
    super().__init__()
  def sample_towards(self, points, sampler): raise NotImplementedError()
  def sample_direction(self, it, sampler, active=True): raise NotImplementedError()
  def intersect(self, _rays): return None, False

# Point light which has constant, linear, and quadratic decay as well as a single coloration.
class PointLights(Light):
  def __init__(
      self,
      intensity=[1., 1., 1.],
      location=[0, 1, 0],
      const=1e-8, linear=1e-8, square=1,
      scale=1e2,
      device = "cuda",
  ):
    super().__init__()
    self.device = device
    self.scale = torch.tensor(scale, dtype=torch.float, requires_grad=True)
    if type(intensity) is torch.Tensor: self.intensity = intensity
    else:
      assert(type(intensity) is type([]))
      self.intensity = torch.tensor(
        [intensity], device=device, requires_grad=True, dtype=torch.float,
      )

    if type(location) is torch.Tensor: self.location = location
    else:
      assert(type(location) is type([]))
      self.location = torch.tensor(
        location, device=device, requires_grad=True, dtype=torch.float,
      )
      if len(self.location.shape) == 1:
        self.location = self.location.unsqueeze(0).detach()
    self.const  = torch.tensor( const, dtype=torch.float, requires_grad=True)
    self.linear = torch.tensor(linear, dtype=torch.float, requires_grad=True)
    self.square = torch.tensor(square, dtype=torch.float, requires_grad=True)
  #
  def parameters(self):
    return chain(self.location_parameters(), self.spectrum_parameters())
  def location_parameters(self): return [self.location]
  def spectrum_parameters(self):
    return [self.scale, self.intensity, self.const, self.linear, self.square]
  def intensity_parameters(self):
    return [self.scale, self.const, self.linear, self.square]
  def coefficients(self): return [self.const, self.linear, self.square]
  # sample from points towards light
  def sample_towards(self, points): return F.normalize(points - self.location, dim=-1)
  def envmap(self, p):
    d = p[None, ...] - self.location[:, None, None, :]
    dist = torch.linalg.norm(d, dim=-1, keepdim=True)
    spectrum = self.const.clamp(min=1e-6) + \
               self.linear.clamp(min=1e-6) * dist + \
               self.square.clamp(min=1e-6) * dist.square()
    spectrum = self.scale * F.normalize(self.intensity, dim=-1)/spectrum.clamp(min=1e-6)
    return spectrum
  def sample_direction(self, it, sampler, active=True):
    ds = DirectionSample()
    ds.p = self.location[:, None, None, None, :]
    # Surface normal
    ds.n = 0
    ds.uv = 0
    ds.obj = self
    ds.delta = True
    ds.d = ds.p - it.p

    # from interaction to self
    ds.dist = torch.linalg.norm(ds.d, dim=-1, keepdim=True)
    #ds.dist = ds.d.square().sum(dim=-1, keepdim=True).clamp(min=1e-7).sqrt()
    ds.d = F.normalize(ds.d, eps=1e-6, dim=-1)
    spectrum = self.const.clamp(min=1e-6) + \
               self.linear.clamp(min=1e-6) * ds.dist + \
               self.square.clamp(min=1e-6) * ds.dist.square()
    color = self.intensity[:, None,None,None, :]
    spectrum = self.scale * F.normalize(color, dim=-1)/spectrum.clamp(min=1e-6)
    spectrum[~active] = 0

    return ds, spectrum

# A "constant" light emitter, aka a sphere that wraps some scene and emits light inwrads.
@dataclass
class Constant(Light):
  sphere: Sphere = Sphere(center=[0,0,0], radius=5)
  intensity: torch.Tensor = torch.tensor(0.5, device="cpu")
  def sample_ray(self, sampler):
    v0 = square_to_uniform_sphere(sampler.sample(points.shape[:-1] + (2,)))
    v1 = square_to_cos_hemisphere(sampler.sample(points.shape[:-1] + (2,)))

    r_o = self.sphere.center + v0 * self.sphere.radius
    r_d = from_local(coordinate_system(-v0), v1)
    spectrum = self.intensity * 4 * (math.pi * self.sphere.radius).square()
    return torch.cat([r_o, r_d], dim=-1),

  def sample_direction(self, it, sampler, active=True):
    device = it.device()
    d = F.normalize(
      square_to_uniform_sphere(sampler.sample(it.p.shape[:-1] + (2,))),
      dim=-1,
    )
    dist = 2 * self.sphere.radius

    ds = DirectionSample(
      p = it.p + d*dist,
      n = -d,
      pdf = square_to_uniform_sphere_pdf(d),
      delta = False,
      obj=self,
      d = d,
      dist = dist,
    )

    spectrum = self.intensity/ds.pdf
    return ds, spectrum.expand(it.shape())
  def eval_pdf(self, src_it, dst_it):
    spectrum = self.intensity.reshape_as(src_it.p)
    d = F.normalize(src_it.p - dst_it.p, dim=-1)
    return spectrum, square_to_uniform_sphere_pdf(d)

def identity(x): return x

# models a 5d light field f(p) -> direction + magnitude
# with a constant R,G,B value
class LightField(nn.Module):
  def __init__(self, device="cuda"):
    super().__init__()

    self.light_field_approx = SkipConnMLP(
      in_size = 3, out=3,
      num_layers=10,
      hidden_size=256,
      device=device,
    ).to(device)

    self.color = nn.Parameter(
      torch.tensor([0., 0., 0.], requires_grad=True, dtype=torch.float, device=device),
      requires_grad=True,
    )
    self.preproc = identity
    self.device=device
    self.preproc = identity
    self.postproc = identity
  def sample_towards(self, points, sampler): raise NotImplementedError()
  def sample_direction(self, it, sampler, active=True):
    preproc = getattr(self, 'preproc', identity)
    non_norm_dir = self.light_field_approx(preproc(it.p[active]))
    ds = DirectionSample()
    # Undefined for light field as is
    ds.p = None
    ds.dist = None

    ds.n = 0
    ds.uv = 0
    ds.obj = self

    ds.delta = True
    ds.pdf = torch.ones(it.p.shape[:-1], device=it.p.device, dtype=torch.float)

    ds.d = torch.zeros_like(it.p)
    ds.d[active] = F.normalize(non_norm_dir, eps=1e-6, dim=-1).clamp(min=1e-6, max=1)
    magn = torch.linalg.norm(non_norm_dir, ord=2, dim=-1, keepdim=True)
    spectrum = torch.zeros_like(it.p, dtype=torch.float)
    spectrum[active] = magn * self.color.sigmoid()
    return ds, spectrum
