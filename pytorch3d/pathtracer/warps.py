import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .neural_blocks import ( DensityEstimator )

@torch.jit.script
def circ(x): return torch.sqrt((1 - x.square()).clamp(min=1e-10))

@torch.jit.script
def square_to_uniform_disk_concentric(sample):
  v = 2*sample - 1
  is_zero = (v == 0).all(dim=-1)
  quadrant_1_or_3 = (v[..., 0].abs() < v[..., 1].abs()).unsqueeze(-1)
  x, y = torch.split(v, 1, dim=-1)

  r = torch.where(quadrant_1_or_3, y, x)
  rp = torch.where(quadrant_1_or_3, x, y)

  # Maybe this is better(but have to find way to keep r from above):
  # r = x/y
  # torch.reciprocal(r[quadrant_1_or_3], out=r[quadrant_1_or_3])

  r = r.sign() * r.abs().clamp(min=1e-12)
  phi = 0.25 * math.pi * rp/r
  phi = torch.where(quadrant_1_or_3, 0.5 * math.pi - phi, phi)
  phi = torch.where(is_zero.unsqueeze(-1), torch.zeros_like(phi), phi)

  s, c = phi.sin(), phi.cos()
  return torch.cat([r * s, r * c], dim=-1)

#@torch.jit.script
def square_to_uniform_sphere(sample):
  #assert(sample.shape[-1] == 2)
  z = 1 - 2 * sample[..., 1]
  r = circ(z)
  tmp = 2 * math.pi * sample[..., 0] - math.pi
  return torch.stack([
    r * tmp.cos(), r * tmp.sin(), z,
  ], dim=-1)

def square_to_uniform_sphere_pdf(sample): return 1/(4 * math.pi)

@torch.jit.script
def square_to_cos_hemisphere(sample):
  #assert(sample.shape[-1] == 2)
  p = square_to_uniform_disk_concentric(sample)
  z = (1 - (p * p).sum(dim=-1, keepdim=True)).clamp(min=1e-7).sqrt()
  return torch.cat([p, z], dim=-1)

@torch.jit.script
def square_to_cos_hemisphere_pdf(d): return d[..., 2]/math.pi

#@torch.jit.script
def random_on_sphere(batches:int, device="cuda"):
  uv = torch.rand(batches, 2, device=device)
  u, v = uv.split(1, dim=-1)
  theta = 2 * math.pi * u
  phi = (2 * v - 1).clamp(min=-1, max=1).acos()
  return torch.cat([
    theta.sin() * phi.cos(),
    theta.sin() * phi.sin(),
    theta.cos(),
  ], dim=-1), uv

class NeuralWarp(nn.Module):
  def __init__(self, device="cuda"):
    super().__init__()
    self.estim = DensityEstimator(device=device)
  def forward(self, shape):
    from .utils import ( uv_to_dir, )
    val, pdf = self.estim(shape)
    return uv_to_dir(val.tanh()), pdf
  def pdf(self, val): return self.estim.pdf(val)
  def prime(
    self, silent=False, lr=1e-3, weight_decay=0,
    iters=10_000, batches=2<<13,
    device="cuda",
    compare_to=square_to_cos_hemisphere_pdf,
  ):
    from tqdm import trange
    import torch.nn.functional as F
    opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
    it = range(iters) if silent else trange(iters)
    update = lambda _: None
    if not silent: update = lambda l: it.set_postfix(refresh=False, loss=f"{l:.05}")

    for i in it:
      opt.zero_grad()
      rand, uv = random_on_sphere(batches, device)
      est_pdf = self.estim.pdf(uv.unsqueeze(1))
      real_pdf = compare_to(rand)
      loss = F.binary_cross_entropy(est_pdf, real_pdf)#.clamp(min=1e-9).sqrt()
      loss.backward()
      opt.step()
      update(loss.item())


class MipMap(nn.Module):
  def __init__(
    self,
    device="cuda",
    depth=4,
  ):
    super().__init__()
    self.percents = nn.Parameter(
      torch.randn(2**depth, device=device, dtype=torch.float, requires_grad=True),
      requires_grad=True,
    )

    self.depth=depth
  def _partition_v(self, samples: ["*", 2], percents, x_max, x_min, y_max, y_min):
    l = len(percents)
    if l == 1: return samples
    top, bot = percents.split(l//2)
    assert(len(top) == len(bot))
    ts = top.sum()
    shift_up = ts/(ts + bot.sum())
    dy = (y_max + y_min)/2
    ys = samples[1]
    ys = torch.where(
      ys >= shift_up,
      dy/(1-shift_up) * (ys - shift_up) + dy,
      ys * dy/shift_up + y_min
    )
    new_samples = torch.stack([samples[0], ys], dim=-1)
    m = ys < dy
    # TODO below swaps the order, maybe change this to where without masking
    # this shouldn't be used right now anyway so it doesn't matter
    return torch.cat([
      self._partition_h(new_samples[m], top, x_max, x_min, y_max, dy),
      self._partition_h(new_samples[~m], bot, x_max, x_min, dy, y_min),
    ], dim=-1)
  def _partition_h(self, samples: ["*", 2], percents, x_max, x_min, y_max, y_min):
    l = len(percents)
    if l == 1: return samples
    left, right = percents.split(l//2)
    assert(len(left) == len(right))
    ls = left.sum()
    shift_left = ls/(ls + right.sum())
    dx = (x_max + x_min)/2
    xs = samples[0]
    xs = torch.where(
      xs >= shift_left,
      dx/(1-shift_left) * (xs - shift_left) + dx,
      xs * dx/shift_left + x_min
    )
    new_samples = torch.stack([xs, samples[1]], dim=-1)
    m = xs < dx
    return torch.cat([
      self._partition_v(new_samples[m], top, x_max, x_dx, y_max, y_min),
      self._partition_v(new_samples[~m], bot, x_dx, x_min, y_max, y_min),
    ], dim=-1)
  # necessary to handle tensors here, because we're processing multiple items at the same time
  def classify_v(self, samples, i_max, i_min, x_max, x_min, y_max, y_min, r_depth):
    if r_depth == 0:
      assert(((i_max - i_min) == 1).all())
      return i_min
    dy = (y_max + y_min)/2
    mi = (i_max + i_min)//2
    ys = samples[..., 1]
    # where y > dy, move the lower index up
    return torch.where(
      ys > dy,
      self.classify_h(samples, i_max, mi, x_max, x_min, y_max, dy, r_depth-1),
      self.classify_h(samples, mi, i_min, x_max, x_min, dy, y_min, r_depth-1),
    )
  def classify_h(self, samples, i_max, i_min, x_max, x_min, y_max, y_min, r_depth):
    if r_depth == 0:
      assert((i_max == i_min).all())
      return i_min
    dx = (x_max + x_min)/2
    mi = (i_max + i_min)//2
    xs = samples[..., 0]
    # where y > dy, move the lower index up
    return torch.where(
      xs > dx,
      self.classify_v(samples, i_max, mi, x_max, dx, y_max, y_min, r_depth-1),
      self.classify_v(samples, mi, i_min, dx, x_min, y_max, y_min, r_depth-1),
    )
  def pdf(self, val):
    assert(((val <= 1) & (val >= -1)).all())
    dev=val.device
    idxs = self.classify_v(
      (val+1)/2,

      torch.full(val.shape[:-1], len(self.percents), dtype=torch.long, device=dev),
      torch.zeros(val.shape[:-1], dtype=torch.long, device=dev),

      torch.ones(val.shape[:-1], device=dev),
      torch.zeros(val.shape[:-1], device=dev),

      torch.ones(val.shape[:-1], device=dev),
      torch.zeros(val.shape[:-1], device=dev),

      self.depth
    )

    return F.softmax(self.percents,dim=-1)[idxs].squeeze(-1)

  def forward(self, shape):
    from .utils import ( uv_to_dir, )
    # TODO implement this by sampling and then partitioning and finding
    raise NotImplementedError()
    val, pdf = self.estim(shape)
    return uv_to_dir(val.tanh()), pdf

  def parameters(self): return [self.percents]

# https://cs.dartmouth.edu/wjarosz/publications/jarosz09importance.pdf
class SphericalHarmonics(nn.Module):
  def __init__(
    self,
    device="cuda",
    order=3,
  ):
    super().__init__()
    self.order = order
    self.coeffs = [
      torch.zeros(2 * l, device=device, dtype=torch.float) for l in range(order)
    ]
    self.device = device
  # sample from this harmonic in a given shape
  def sum_of_integrals(self):
    s = self.coeffs[0] * todo
    for i in range(1, self.order):
      coeffs = self.coeffs[i]
      l = len(coeffs)
      s += coeffs * todo
  def sph_harmonic_int_phi(self):
    ...
  def sph_harmonic_int_theta(self):
    ...
  def forward(self, shape):
    from .utils import ( uv_to_dir, )
    # generate random uniform with shape
    rnd = torch.rand(shape, dtype=torch.float, device=self.device)
    tform, pdf = self.transform(rnd)
    return uv_to_dir(val.tanh()), pdf
  def pdf(self, val): return self.estim.pdf(val)


# Computes the legendre polynomial evaluations at x up to a given order
def legendre(x, up_to_order):
  out = [torch.tensor(1, dtype=x.dtype, device=x.device), x]
  for n in range(2, up_to_order):
    out.append(
      ((2 * n + 1) * x * out[-1] - a * out[-2])/(n+1)
    )
  return torch.stack(out)

def assoc_legendre_integral(order, a_evals, b_evals, a, b):
  raise NotImplementedError()
