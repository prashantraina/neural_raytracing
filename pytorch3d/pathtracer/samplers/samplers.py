import torch
import torch.nn as nn

class Sampler(nn.Module):
  # Default independent sampler class
  # Should generally pick a different sampler implementation but this one works ok
  def __init__(self, sample_count=0, device="cuda"):
    super().__init__()
    self.sample_count = sample_count
    self.dimension_index = 0
    self.sample_index = 0
    self.samples_per_wavefront = 1
    self.wavefront_size = 0
    self.base_seed = 0
    self.device = device

  # returns samples in the shape of `shape`
  def sample(self, shape, device=None):
    if device is None: device = self.device
    return torch.rand(shape, device=device)

  def current_sample_index(self, size):
    wavefront_sample_offsets = torch.zeros(size)
    if self.samples_per_wavefront > 1:
      wavefront_samples_offsets = torch.arange(size) % self.samples_per_wavefront
    return self.sample_index * self.samples_per_wavefront + wavefront_sample_offsets

def _is_prime(x):
  for i in range(2, x // 2 + 1):
    if x % i == 0: return False
  return True

def encode_all_ones(x):
  x |= x >> 1
  x |= x >> 2
  x |= x >> 4
  x |= x >> 8
  x |= x >> 16
  return x

def nearest_power_of_two(x): return encode_all_ones(x-1)+1


# defines a permutation vector over idx up to samples
# See: http://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf
# Implementation from:
# https://github.com/mitsuba-renderer/mitsuba2/blob/master/include/mitsuba/core/random.h
def permute_kensler(idx: [int], samples: int, seed: int) -> torch.Tensor:
  if samples == 1: return torch.Tensor(0).int()
  w = torch.ones(idx.shape).int() * samples - 1
  w = encode_all_ones(w)
  max_iter = nearest_power_of_two(samples) - samples + 1
  valid = torch.ones(idx.shape).bool()
  for i in range(max_iter):
    tmp = idx.clone()
    tmp ^= seed
    tmp *= 0xe170893d
    tmp ^= seed >> 16
    tmp ^= (tmp & w) >> 4
    tmp ^= seed >> 8
    tmp *= 0x0929eb3f
    tmp ^= seed >> 23
    tmp ^= (tmp & w) >> 1
    tmp *= 1 | seed >> 27
    tmp *= 0x6935fa69
    tmp ^= (tmp & w) >> 11
    tmp *= 0x74dcb303
    tmp ^= (tmp & w) >> 2
    tmp *= 0x9e501cc3
    tmp ^= (tmp & w) >> 2
    tmp *= 0xc860a3df
    tmp &= w
    tmp ^= tmp >> 5
    idx[valid] = tmp
    if not valid.any(): break
  return (idx + seed) % samples

class OrthogonalSampler(Sampler):
  def __init__(
    self,
    sample_count,
    jitter=True,
    strength=2,
  ):
    super().__init__(sample_count)
    self.jitter = jitter
    self.strength = strength
    assert(strength == 2), "does not support other strengths yet"

    self.dimension_index = 0

    resolution = 2
    sqr = lambda x: x * x
    while (sqr(resolution) < self.sample_count and not is_prime(resolution)):
      m_resolution += 1
    self.sample_count = resolution
    self.resolution = resolution
  def sample(self, shape): return self.next_1d(torch.prod(shape)).reshape(shape)
  def next_1d(self, dim):
    out = self.bose(
      self.current_sample_index(),
      self.dimension_index,
      self.base_seed,
    )
    self.dimension_index += 1
    return out
  def next_2d(self, dim):
    return torch.cat([
      self.next_1d(dim).unsqueeze(-1),
      self.next_1d(dim).unsqueeze(-1),
    ], dim=-1)
  # Bose construction technique for orthogonal arrays. It only support OA of strength == 2
  def bose(self, i, dim, permute_seed):
    i = permute_kensler(i % self.sample_count, self.sample_count, permute_seed);

    a_i0 = i // self.resolution
    a_i1 = i - a_i0 * self.resolution
    a_ij, a_ik = a_i0, a_i1
    if dim == 0:
      ...
    elif dim == 1:
      a_ij, a_ik = a_ik, a_ij
    else:
      k = dim -1 if dim % 2 == 0 else dim + 1
      a_ij = (a_i0 + (dim - 1) * a_i1) % self.resolution;
      a_ik = (a_i0 + (k - 1) * a_i1) % self.resolution;

    stratum = permute_kensler(a_ij, m_resolution, p * (j + 1) * 0x51633e2d, active);
    sub_stratum = permute_kensler(a_ik, m_resolution, p * (j + 1) * 0x68bc21eb, active);
    jitter = torch.rand(i.shape)  if self.jitter else 0.5
    return (stratum + (sub_stratum + jitter) / self.resolution) / self.resolution;
  def bush(self, i, dim, permute_seed):
    N = self.resolution ** self.strength
    raise NotImplementedError
