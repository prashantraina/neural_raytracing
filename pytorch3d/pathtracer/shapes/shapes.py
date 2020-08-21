import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

from ..interaction import SurfaceInteraction

EPS = 1e-8

def quad_solve(a, b, c):
  d = b*b - 4 * a * c
  # This could also
  valid = d > 0 # d is discriminant
  #assert(d[valid].isfinite().all())
  d[valid] = d[valid].sqrt()
  s = torch.stack([d, -d], dim=-1)
  return (-b[..., None] + s)/(2*a[..., None]), valid

@dataclass
class Shape(nn.Module):
  def __init__(self):
    super().__init__()
  def intersect(self, rays, max_t=math.inf, active=True): raise NotImplementedError()
  def intersect_test(self, rays, max_t = math.inf, active=True):
    return self.intersect(rays, max_t=max_t, active=active)[1]
  # returns the lower and upper bounds of intersection for this shape.
  # used when using this shape to bound another
  def intersect_limits(self, rays, max_t=math.inf, active=True): raise NotImplementedError()

class Sphere(Shape):
  center: torch.Tensor
  radius: float
  device: torch.device
  def __init__(self, center, radius, device="cuda"):
    self.device = torch.device(device)
    self.center = torch.tensor(center, device=device, dtype=torch.float)
    self.radius = float(radius)
    self.sqr_radius = self.radius * self.radius

  # For now can only represent 1 sphere at a time
  def __len__(self): return 1

  def intersect(self, rays, active=True, primary=True):
    r_o, r_d = torch.split(rays, 3, dim=-1)
    fs = r_o - self.center
    a = torch.sum(r_d * r_d, dim=-1)
    b = 2 * torch.sum(r_d * fs, dim=-1)
    c = torch.sum(fs*fs, dim=-1) - self.sqr_radius
    intersections, mask = quad_solve(a, b, c)
    # if either intersection point is > EPS it's valid.
    mask = mask & (intersections >= EPS).any(-1)
    # zero out intersections behind camera
    intersections[intersections < EPS] = math.inf
    # find minimum intersections
    t, _ = intersections.min(dim=-1)
    p = r_o + t[..., None] * r_d
    n = F.normalize(p - self.center, dim=-1)
    p += n * 1e-5
    si = SurfaceInteraction(
      p=p,
      t=t,
      obj=self,
    )
    si.set_normals(n)
    si.wi = si.to_local(-r_d)
    return si, mask
  def intersect_test(self, rays, active=True):
    r_o, r_d = torch.split(rays, 3, dim=-1)
    fs = r_o - self.center
    a = torch.sum(r_d * r_d, dim=-1)
    b = 2 * torch.sum(r_d * fs, dim=-1)
    c = torch.sum(fs*fs, dim=-1) - self.sqr_radius
    intersections, mask = quad_solve(a, b, c)
    return mask & (intersections >= EPS).any(-1)
  def intersect_limits(self, rays, max_t=math.inf, active=True):
    r_o, r_d = torch.split(rays, 3, dim=-1)
    fs = r_o - self.center
    a = torch.sum(r_d * r_d, dim=-1)
    b = 2 * torch.sum(r_d * fs, dim=-1)
    c = torch.sum(fs*fs, dim=-1) - self.sqr_radius
    intersections, mask = quad_solve(a, b, c)
    # if either intersection point is > EPS it's valid.
    mask = mask & (intersections >= EPS).any(-1)
    # zero out intersections behind camera
    intersections[intersections < EPS] = math.inf
    # find minimum intersections
    lower, _ = intersections.min(dim=-1)
    upper, _ = intersections.max(dim=-1)
    return lower, upper, mask
  def uv(self, p):
    print(sphere.center.shape, p.shape)
    exit()
    d_x, d_y, d_z = (sphere.center - p).split(1, dim=-1)
    u = 0.5 + torch.atan2(d_x, d_z)/(2 * math.pi)
    v = 0.5 - d_y.asin()/math.pi
    return torch.cat([u, v], dim=-1)

class SphereCloud(Shape):
  centers: torch.Tensor # [N_spheres, 4(center x,y,z,radius)]
  radii: torch.Tensor
  device: torch.device
  def __init__(self, centers=[[0,0,0]], radii=1, device="cuda"):
    self.device = torch.device(device)

    N = len(centers)
    self.centers = torch.zeros([N, 3], dtype=torch.float, device=self.device)
    for i in range(N): self.centers[i] = torch.tensor(centers[i], device=device)

    self.radii = torch.full([N], radii, dtype=torch.float, device=self.device)

  def __len__(self): return 1
  def intersect(self, rays, active=True, t_max = math.inf, split_n=256):
    device=self.device
    r_o, r_d = torch.split(rays, 3, dim=-1)

    out_active = torch.zeros(r_o.shape[:-1], dtype=torch.bool, device=device)

    best_dists = torch.full_like(out_active, t_max, dtype=torch.float, device=device)
    best_faces = torch.full_like(out_active, -1, dtype=torch.long, device=device)

    r_o_r = r_o.expand(split_n, *r_o.shape)
    r_d_r = r_d.expand_as(r_o_r)

    iters = zip(self.centers.split(split_n, dim=0), self.radii.split(split_n, dim=0))
    for spheres, radii in iters:
      batch_size = spheres.shape[0]
      # [batch_size, W, H, BundleSize, 3], " "
      r_o_r, r_d_r = r_o_r[:batch_size], r_d_r[:batch_size]

      # [batch_size, W, H, BundleSize, 4]
      sphere_exp = spheres[:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)\
        .transpose(0, 1)
      radii_exp = radii[:, None, None, None].repeat(1, *r_d_r.shape[1:-1])\
        .transpose(0, 1)

      fs = r_o_r - sphere_exp

      a = torch.sum(r_d_r * r_d_r, dim=-1)
      b = 2 * torch.sum(r_d_r * fs, dim=-1)
      c = torch.sum(fs*fs, dim=-1) - (radii_exp * radii_exp)
      intersections, mask = quad_solve(a, b, c)

      mask = mask & ((intersections >= EPS) & (intersections < t_max)).any(-1)
      intersections[intersections < EPS] = math.inf

      valid_mins = mask.any(0)

      # TODO might need to convert this into = out_active | valid_mins
      out_active = out_active | valid_mins

      # find minimum intersections for each sphere
      t, _ = intersections.min(dim=-1)
      t[~mask] = math.inf
      min_t, sph_idx = t.min(dim=0)
      lesser = best_dists > min_t
      replace_cond = valid_mins & lesser

      best_dists[replace_cond] = min_t[replace_cond]
      best_faces[replace_cond] = sph_idx[replace_cond]

    # compute positions and all
    p = r_o + best_dists[..., None] * r_d
    n = torch.zeros_like(p, dtype=torch.float, device=device)
    n[out_active] = F.normalize(
      p[out_active] - self.centers[best_faces[out_active]],
      dim=-1,
    )
    p += n * 1e-5
    si = SurfaceInteraction(
      p=p,
      t=best_dists,
      obj=self,
    )
    si.set_normals(n)
    si.wi=si.to_local(-r_d)
    return si, out_active
  def intersect_test(self, rays, active=True, t_max = math.inf, split_n=256):
    device=self.device
    r_o, r_d = torch.split(rays, 3, dim=-1)

    out_active = torch.zeros(r_o.shape[:-1], dtype=torch.bool, device=device)

    r_o_r = r_o.expand(split_n, *r_o.shape)
    r_d_r = r_d.expand_as(r_o_r)

    iters = zip(self.centers.split(split_n, dim=0), self.radii.split(split_n, dim=0))
    for spheres, radii in iters:
      batch_size = spheres.shape[0]
      # [batch_size, W, H, BundleSize, 3], " "
      r_o_r, r_d_r = r_o_r[:batch_size], r_d_r[:batch_size]

      # [batch_size, W, H, BundleSize, 4]
      sphere_exp = spheres[:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)

      fs = r_o_r - sphere_exp

      a = torch.sum(r_d_r * r_d_r, dim=-1)
      b = 2 * torch.sum(r_d_r * fs, dim=-1)
      c = torch.sum(fs*fs, dim=-1) - (radii * radii)[..., None]
      intersections, mask = quad_solve(a, b, c)

      mask = mask & ((intersections >= EPS) & (intersections < t_max)).any(-1)
      out_active = out_active | mask.any(0)

    return out_active
