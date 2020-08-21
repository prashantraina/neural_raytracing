import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .shapes import Shape
from ..interaction import ( SurfaceInteraction, MixedInteraction )
from pytorch3d.pathtracer.utils import ( fourier, create_fourier_basis, smooth_min )
from ..neural_blocks import ( SkipConnMLP )
from tqdm import trange
import random

# Default unit sphere SDF.
SPHERE_SDF = lambda p: torch.norm(p, dim=-1) - 1

# A set of spheres which are smooth-min-ed together and a residual MLP.
class SphereSDF(nn.Module):
  def __init__(self, n=2<<6, device="cuda"):
    super().__init__()
    self.centers = nn.Parameter(0.3 * torch.rand(n,3,device=device, requires_grad=True) - 0.15)
    self.radii = nn.Parameter(0.2 * torch.rand(n, device=device, requires_grad=True) - 0.1)

    self.tfs = nn.Parameter(torch.zeros(n, 3, 3, device=device, requires_grad=True))
    self.shift = SkipConnMLP(
      num_layers=8,
      hidden_size=128,
      in_size=3,out=1,
      device=device,
      freqs=32,
      activation=F.softplus,
      zero_init=True,
    ).to(device)

  def set_center(self, at):
    self.centers = nn.Parameter(at.expand_as(self.centers).clone().detach())

  @torch.jit.export
  def transform(self, p):
    tfs = self.tfs + torch.eye(3, device=p.device).unsqueeze(0)
    return torch.einsum("ijk,ibk->ibj", tfs, p.expand(tfs.shape[0], -1, -1))
  def forward(self, p):
    q = self.transform(p.reshape(-1, 3).unsqueeze(0)) - self.centers.unsqueeze(1)
    sd = q.norm(p=2, dim=-1) - self.radii.unsqueeze(-1)
    out = smooth_min(sd, k=32.).reshape(p.shape[:-1])
    return out + self.shift(p).reshape_as(out)

# Same as above except for boxes with rounded edges without rotation and no residual.
# Can be used for experimentation.
class RoundBoxSDF(nn.Module):
  def __init__(self, n=2<<4, device="cuda"):
    super().__init__()
    self.centers = nn.Parameter(0.3 * torch.rand(n,3,device=device, requires_grad=True) - 0.15)
    self.b = nn.Parameter(0.2 * torch.rand_like(self.centers, requires_grad=True))
    self.radii = nn.Parameter(0.2 * torch.rand(n, device=device, requires_grad=True) - 0.1)

    self.tfs = nn.Parameter(torch.zeros(n, 3, 3, device=device, requires_grad=True))

  @torch.jit.export
  def transform(self, p):
    tfs = self.tfs + torch.eye(3, device=p.device).unsqueeze(0)
    return torch.einsum("ijk,ibk->ibj", tfs, p.expand(tfs.shape[0], -1, -1))
  def forward(self, p):
    q = (self.transform(p.reshape(-1, 3).unsqueeze(0))
      - self.centers.unsqueeze(1)).abs() - self.b.unsqueeze(1)
    up = q.clamp(min=1e-7).norm(p=2, dim=-1, keepdim=True)
    x, y, z = q.split(1, dim=-1)
    down = torch.maximum(x, torch.maximum(y, z)).clamp(max=-1e-7)
    sd = up + down
    return smooth_min(sd, k=16.).reshape(p.shape[:-1])

# Same as above except for capsules and no residual.
# Can be used for experimentation.
class CapsuleSDF(nn.Module):
  def __init__(self, n=2<<5, device="cuda"):
    super().__init__()
    self.a = nn.Parameter(0.1 * torch.rand(n,3,device=device, requires_grad=True) - 0.05)
    self.b = nn.Parameter(0.1 * torch.rand_like(self.a, requires_grad=True)- 0.05)
    self.radii = nn.Parameter(0.1 * torch.rand(n, device=device, requires_grad=True) - 0.05)
  def forward(self, p):
    pa = p.reshape(-1, 3).unsqueeze(0) - self.a.unsqueeze(1)
    ba = (self.b - self.a).unsqueeze(1)
    h = (pa * ba).sum(dim=-1, keepdims=True)/\
        (ba * ba).sum(dim=-1, keepdims=True)\
        .clamp(min=0, max=1)
    sd = (pa - ba * h).norm(p=2, dim=-1) - \
         self.radii.unsqueeze(-1)
    return smooth_min(sd, k=16.).reshape(p.shape[:-1])

# A general SDF class, which takes an arbitrary SDF and performs ray-marching for intersection.
class SDF():
  device: torch.device
  def __init__(
    self,
    device="cuda",
    sdf = SPHERE_SDF,
    epsilon = 1e-3,
    max_steps = 32,
    dist=2.2,
    **kwargs,
  ):
    #super().__init__()
    self.device = torch.device(device)

    self.sdf = sdf
    self.epsilon = epsilon
    self.max_steps = max_steps
    self.dist = dist

  def __len__(self): return 1
  def parameters(self): return self.sdf.parameters()
  # Check for intersection with this SDF as well as compute the throughput.
  def intersect(self, rays, max_t=10, active=True, primary: bool=True):
    device = self.device
    r_o, r_d = rays.split(3, dim=-1)
    depths = torch.zeros(r_o.shape[:-1] + (1,), dtype=torch.float, device=device)
    remaining = torch.ones(depths.shape[:-1], dtype=torch.bool, device=device)
    out_active = torch.zeros_like(remaining, dtype=torch.bool, device=device)

    with torch.no_grad():
      for i in range(self.max_steps):
        remaining = remaining & (depths < max_t).squeeze(-1)
        dists = self.sdf(r_o + r_d * depths)
        hits = remaining & (dists <= self.epsilon)
        # any item with distance less than epsilon is active
        out_active = out_active | hits

        remaining = remaining & ~hits
        depths = torch.where(
          remaining.unsqueeze(-1),
          depths + dists.unsqueeze(-1),
          depths,
        )

    p = r_o + depths * r_d
    throughput = 0
    alpha = 1000
    if primary:
      throughput, best_pos = self.throughput(r_o, r_d)
      throughput = -alpha * throughput

      #missed = (~out_active) & (throughput.sigmoid() > 0.8)
      #p = torch.where(
      #  missed.unsqueeze(-1),
      #  best_pos,
      #  p,
      #)
      #out_active = out_active | missed
    si = MixedInteraction(
      p=p, t=depths.squeeze(),
      obj=self,
      throughput=throughput,
    )
    normals = torch.zeros_like(p, dtype=torch.float, device=device)
    if out_active.any():
      normals_active = self.autograd_diff(p[out_active])
      setattr(si, "raw_normals", normals_active)
      normals[out_active] = F.normalize(normals_active, eps=1e-6, dim=-1)
      p[out_active] = p[out_active] + normals[out_active] * self.epsilon * 5
    si.set_normals(normals)
    si.wi = si.to_local(-r_d)
    return si, out_active
  # test for intersection with this SDF.
  def intersect_test(self, rays, max_t=10, active=True):
    device = self.device
    r_o, r_d = rays.split(3, dim=-1)
    depths = torch.zeros(r_o.shape[:-1] + (1,), dtype=torch.float, device=device) + \
      1e2 * self.epsilon
    remaining = torch.ones(depths.shape[:-1], dtype=torch.bool, device=device)

    with torch.no_grad():
      for i in range(self.max_steps):
        #remaining = remaining & (depths < max_t).squeeze(-1)
        dists = self.sdf(r_o + r_d * depths)
        hits = remaining & (dists < self.epsilon)

        depths = torch.where(
          remaining.unsqueeze(-1),
          depths + dists.unsqueeze(-1),
          depths,
        )
        remaining = remaining & ~hits
    return (depths >= max_t).squeeze(-1) | remaining

  # Autograd used for computing normals, which is essentially the same as IDR.
  def autograd_diff(self, p):
    with torch.enable_grad():
      if not p.requires_grad: p = p.requires_grad_()
      out = self.sdf(p)
      grad_outputs = torch.ones_like(out)
      n, = torch.autograd.grad(
        inputs=p,
        outputs=out,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
      )
      return n
  # Forward differences approximation for SDFs. Is not numerically stable enough for learning.
  def forward_differences(self, points):
    at = self.sdf(points)
    return F.normalize(torch.stack([
      self.sdf(points + self.fd_x) - at,
      self.sdf(points + self.fd_y) - at,
      self.sdf(points + self.fd_z) - at,
    ], dim=-1), dim=-1)
  # "primes" this learned SDF to a known_sdf, to be done prior to training to learn a mesh
  # in order to have a coherent SDF at the start.
  def prime(
    self,
    opt, known_sdf,
    loss_fn = F.mse_loss,
    min=-5, max=5, batch_size=4096, iters=5000, ok_eps=1e-6, silent=False,
    device="cuda",
  ):
    t = (range if silent else trange)(iters)
    update = lambda _: None
    if not silent:
      update = lambda loss: t.set_postfix(refresh=False, loss=f"{loss:.06}")
    span = max - min
    for i in t:
      opt.zero_grad()
      p = min + torch.rand(batch_size, 3, device=device)*span
      expected = known_sdf(p)
      got = self.sdf(p)
      loss = loss_fn(expected, got)
      loss.backward()
      opt.step()
      loss = loss.item()
      if loss < ok_eps: break
      update(loss)

  def throughput(self, r_o_local, d):
    batch_size = 128
    # some random jitter I guess?
    dist = getattr(self, 'dist', 2.2)
    max_t = dist+random.random()*(2/batch_size)
    step = max_t/batch_size
    with torch.no_grad():
      sd = self.sdf(r_o_local).squeeze(-1)
      curr_min = sd
      idxs = torch.zeros_like(sd, dtype=torch.long, device=d.device)
      for i in range(batch_size):
        t = step * (i+1)
        sd = self.sdf(r_o_local + t * d).squeeze(-1)
        idxs = torch.where(sd < curr_min, i+1, idxs)
        curr_min = torch.minimum(curr_min, sd)
    idxs = idxs.unsqueeze(-1)
    best_pos = r_o_local  + idxs.unsqueeze(-1) * step * d
    return self.sdf(best_pos), best_pos
  def half_res_throughput(self, r_o_local, d):
    r_o_half = r_o_local[:, ::2, ::2, ...]
    d_half = d[:, ::2, ::2, ...]
    throughput = self.throughput(r_o_half, d_half)
    return throughput.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
  # moderately faster but still needs multiple steps
  def batch_throughput(self, r_o_local, d):
    batch_size = 56
    per = 32
    max_t = 2
    with torch.no_grad():
      ts = torch.linspace(0, max_t, batch_size + random.randint(0, 8), device=d.device)
      bests = None
      b_idxs = None
      for tb in ts.split(per):
        mins, idxs = torch.min(
          self.sdf(r_o_local.unsqueeze(0) + torch.tensordot(tb, d, dims=0)),
          dim=0
        )
        if b_idxs is None:
          b_idxs = idxs
          bests = mins
        else:
          mask = mins<bests
          b_idxs = torch.where(mask, idxs, b_idxs)
          bests[mask] = mins[mask]
      best_t = ts[b_idxs]
    return self.sdf(r_o_local + best_t.unsqueeze(-1) * d)

# an SDF over some number of boxes (was used while experimenting) now defunct.
def Box(sizes):
  def box_sdf(p):
    q = p.abs() - sizes
    return torch.norm(q.clamp(min=0), dim=-1) \
      + torch.max(torch.max(q[..., 0], q[..., 1]), q[..., 2]).clamp(max=0)
  return box_sdf

# Basic sphere SDF
def _sphere_sdf(local, rads):
  lens = torch.norm(local, dim=-1)
  return lens - rads[:, None, None, None]
# Basix AABB SDF
def _box_sdf(local, sizes=0.5):
  q = local.abs() - sizes
  return torch.norm(q.clamp(min=0), dim=-1) \
    + torch.max(torch.max(q[..., 0], q[..., 1]), q[..., 2]).clamp(max=0)
# Basic capsule (pill shaped) SDF.
def _capsule_oriented_sdf(local, desc):
  a, b, r = desc.split([3, 3, 1], dim=-1)
  pa = local - a[:, None, None, None]
  ba = (b - a)[:, None, None, None].expand_as(pa)
  h = ((pa * ba).sum(dim=-1, keepdim=True)/(ba * ba).sum(dim=-1, keepdim=True)).clamp(0, 1)
  v = torch.norm(pa - ba * h, dim=-1)
  return v - r[: None, None, None]

# Compute the smooth min (used below)
def exp_smooth_min(tensor, k=32, dim=-1):
  return torch.where(
    tensor.isfinite().all(0),
    -(torch.exp(-k * tensor).sum(dim).abs()+1e-7).log()/k,
    tensor.min(dim)[0].squeeze()
  )

# A number of SDFs all parametrized together so they can be jointly optimized.
# Was used in initial experimentations
class ParametricSDFSet():
  def __len__(self): return 1
  def __init__(
    self,
    # some strange default so it's immediately recognizable.
    num_shapes=5,
    device="cuda",
  ):
    self.num_shapes = num_shapes

    # bias initially towards 0 center
    self.offsets = torch.rand(num_shapes, 3, device=device)-0.5

    VARIANTS = 3 # number of activated items below
    self.sph_rads = torch.rand(num_shapes, device=device) * 0.3 + 0.1
    self.box_sizes = torch.rand(num_shapes, 3, device=device) * 0.3 + 0.1
    self.capsules = torch.rand(num_shapes, 7, device=device)

    self.descriptors = torch.rand(num_shapes, VARIANTS, device=device)

  def __call__(self, p):
    p_exp = p[None, ...]
    local = p_exp - self.offsets[:, None, None, None]
    normalization = F.softmax(self.descriptors, dim=-1)
    sdfs = torch.stack([
      _sphere_sdf(local, self.sph_rads),
      _box_sdf(local, self.box_sizes[:, None, None, None]),
      _capsule_oriented_sdf(local, self.capsules),
    ], dim=-1)
    per_face_hits = (normalization[:, None, None, None] * sdfs).sum(dim=-1)
    return exp_smooth_min(per_face_hits, dim=0)
  def parameters(self):
    params = [
      self.offsets,
      self.sph_rads,
      self.box_sizes,
      self.capsules,
      self.descriptors,
    ]
    for p in params:
      p.requires_grad = True
    return params
