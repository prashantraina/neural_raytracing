from dataclasses import dataclass
import torch
import torch.nn.functional as F
import math

# https://github.com/mitsuba-renderer/mitsuba2/blob/2fb0c9634e696887fae3921a60818a3a503c892e/include/mitsuba/core/vector.h#L116
# had to be significantly modified in order to add numerical stability while back-propagating.
@torch.jit.script
def coordinate_system(n):
  n = F.normalize(n, eps=1e-7, dim=-1)
  x, y, z = n.split(1, dim=-1)
  sign = torch.where(z >= 0, 1., -1.)
  s_z = sign + z
  a = -torch.where(
    s_z.abs() < 1e-6,
    torch.tensor(1e-6, device=z.device),
    s_z,
  ).reciprocal()
  b = x * y * a

  s = torch.cat([
    (x * x * a * sign) + 1, b * sign, x * -sign,
  ], dim=-1)
  s = F.normalize(s, eps=1e-7, dim=-1)
  t = F.normalize(s.cross(n, dim=-1), eps=1e-7, dim=-1)
  s = F.normalize(n.cross(t, dim=-1), eps=1e-7, dim=-1)
  return torch.stack([s, t, n], dim=-1)

# creates a frame from two vectors
@torch.jit.script
def partial_frame(n, wi):
  c = F.normalize(torch.cross(n, wi, dim=-1), eps=1e-7, dim=-1)
  # TODO might need to handle case where n = wi
  return torch.stack([n, wi, c], dim=-1)

# frame: [..., 3, 3], wo: [..., 3]
@torch.jit.script
def to_local(frame, wo):
  wo = wo.unsqueeze(-1).expand_as(frame)
  out = F.normalize((frame * wo).mean(dim=-2), eps=1e-7, dim=-1)
  return out

# frame: [..., 3, 3], v: [..., 3]
@torch.jit.script
def from_local(frame, v):
  s, t, n = frame.split(1, dim=-1)
  x, y, z = v.split(1, dim=-1)
  wo = s.squeeze(-1) * x + \
    t.squeeze(-1) * y + \
    n.squeeze(-1) * z
  return F.normalize(wo, eps=1e-7, dim=-1)

# Base interaction class
@dataclass
class Interaction():
  p: torch.Tensor
  def spawn_rays(self, d): return torch.cat([self.p.expand_as(d), d], dim=-1)

# SurfaceInteraction represents light interacting with some surface
@dataclass
class SurfaceInteraction(Interaction):
  uv: torch.Tensor = None
  wi: torch.Tensor = None
  t: torch.Tensor = torch.tensor(math.inf, device="cuda")
  bsdf: "BSDF" = None
  obj: "Shape" = None
  bidirectional_normals: bool = False
  # Shading frame
  frame = None

  n: torch.Tensor = None

  def set_normals(self, normals):
    self.n = normals
    self.frame = coordinate_system(normals)
    #assert(self.frame.isfinite().all())

  def to_local(self, wo): return to_local(self.frame, wo)
  def from_local(self, v): return from_local(self.frame, v)

  def shape(self): return self.p.shape
  def device(self): return self.p.device

  @classmethod
  def positions(cls, positions): return cls(positions)

  @classmethod
  def zeros(cls, shape, device):
    return cls(torch.zeros(shape, dtype=torch.float, device=device))

  @classmethod
  def like(cls, tensor): return cls(torch.zeros_like(tensor, dtype=torch.float))

# A SurfaceInteraction which also carries extra data for regression
@dataclass
class MixedInteraction(SurfaceInteraction):
  # represents an interaction with both surfaces and media (used for efficiency instead of
  # keeping two)

  throughput: torch.Tensor = None
  # For some mixed solid/volume interaction, jointly use this struct
  # to encode some gas interaction
  medium_mask = torch.tensor(False) # just store on CPU since this will be overwritten if used
  with_logits: bool = True
  def mark_mediums(self, medium_mask): self.medium_mask = medium_mask
  def surface_interactions(self): return ~self.medium_mask


# A sample over a hemisphere, representing light samples (BSDFs have a separate class in their
# file).
@dataclass
class DirectionSample():
  p: torch.Tensor = None # positions
  n: torch.Tensor = 0 # normals
  pdf: torch.Tensor = 1 # PDF of direction
  delta: torch.Tensor = True # Sampled from dirac delta function
  obj: "Object" = None # which object was this from
  d: torch.Tensor = None # direction
  dist: torch.Tensor = None # distances
