import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
  weak_sigmoid, log_polar_indices,
  cartesian_indices, cartesian_indices_avgs,

  fourier2, create_fourier_basis2,
)
import torch.distributions as D

class SkipConnMLP(nn.Module):
  "MLP with skip connections and fourier encoding"
  def __init__(
    self,

    num_layers = 8,
    hidden_size=64,
    in_size=3,
    out=3,

    skip=3,
    freqs = 16,
    sigma=2<<4,
    device="cuda",
    activation = lambda x: F.leaky_relu(x, inplace=True),

    latent_size=0,

    zero_init = False,
    xavier_init = False,
  ):
    super(SkipConnMLP, self).__init__()
    self.in_size = in_size
    assert(type(freqs) == int)
    self.basis_p, map_size = create_fourier_basis2(
      freqs, features=in_size, freq=sigma, device=device
    )

    self.dim_p = map_size + latent_size
    self.skip = skip
    self.latent_size = latent_size
    skip_size = hidden_size + self.dim_p

    # with skip
    hidden_layers = [
      nn.Linear(
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size,
        hidden_size,
      ) for i in range(num_layers)
    ]

    self.init =  nn.Linear(self.dim_p, hidden_size)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = nn.Linear(hidden_size, out)
    weights = [
      self.init.weight,
      self.out.weight,
      *[l.weight for l in self.layers],
    ]
    biases = [
      self.init.bias,
      self.out.bias,
      *[l.bias for l in self.layers],
    ]
    if zero_init:
      for t in weights: nn.init.zeros_(t)
      for t in biases: nn.init.zeros_(t)
    if xavier_init:
      for t in weights: nn.init.xavier_uniform_(t)
      for t in biases: nn.init.zeros_(t)

    self.activation = activation

  def forward(self, p, latent=None):
    batches = p.shape[:-1]
    init = fourier2(p.reshape(-1, self.in_size), self.basis_p)
    if latent is not None:
      init = torch.cat([init, latent.reshape(-1, self.latent_size)], dim=-1)
    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i != len(self.layers)-1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x))
    out_size = self.out.out_features
    return self.out(self.activation(x)).reshape(batches + (out_size,))
  # sets this MLP to always just return its own input
  def prime_identity(
    self,
    lr=1e-4,
    iters=50_000,
    batches=4096,
    device="cuda",
  ):
    opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0)
    for i in range(iters):
      opt.zero_grad()
      x = torch.rand_like(batches, self.in_size, device=device)
      y = self(x)
      loss = F.mse_loss(x, y)
      loss.backward()
      opt.step()

class TwoStageMLP(nn.Module):
  "Two stage Skip Connected MLP with fourier encoding"
  def __init__(
    self,

    num_layers = 6,
    hidden_size=128,
    in_1=3,
    in_2=3,
    intermediate=1,
    out=3,

    skip=3,
    # same frequencies are used for both inputs
    freqs = [2**4, 2**5, 2**5, 2**5, 2**6, 2**6, 2**6, 2**7, 2**7, 2**8],
    device="cuda",
    activation = F.relu,
  ):
    super(TwoStageMLP, self).__init__()
    self.in_1 = in_1
    self.in_2 = in_2

    n_f = len(freqs)
    self.basis_1 = create_fourier_basis([in_1], freqs, n_f, device)
    self.basis_2 = create_fourier_basis([in_2], freqs, n_f, device)

    self.dim_1 = in_1 + 2 * n_f
    self.skip = skip
    skip_size = hidden_size + self.dim_1

    hidden_layers_1 = [
      nn.utils.weight_norm(nn.Linear(
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size,
        hidden_size,
      )) for i in range(num_layers)
    ]
    self.init =  nn.Linear(self.dim_1, hidden_size)
    self.layers_1 = nn.ModuleList(hidden_layers_1)

    self.inter = nn.Linear(hidden_size, intermediate)

    mid_input_size = intermediate + in_2 + 2 * n_f
    self.from_inter =  nn.Linear(mid_input_size, hidden_size)
    skip_size = hidden_size + mid_input_size
    hidden_layers_2 = [
      nn.utils.weight_norm(nn.Linear(
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size,
        hidden_size,
      )) for i in range(num_layers)
    ]
    self.layers_2 = nn.ModuleList(hidden_layers_2)
    self.out =  nn.Linear(hidden_size, out)

    self.activation = activation

  def forward(self, i1, i2):
    batches = i1.shape[:-1]

    init = fourier(i1.reshape(-1, self.in_1), self.basis_1)
    x = self.init(init)
    for i, layer in enumerate(self.layers_1):
      if i != len(self.layers_1)-1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x, inplace=True))
    x = self.inter(self.activation(x, inplace=True)).sigmoid()
    mid_init = fourier(i2.reshape(-1, self.in_2), self.basis_2)
    mid_init = torch.cat([x, mid_init], dim=-1)
    x = self.from_inter(mid_init)
    for i, layer in enumerate(self.layers_2):
      if i != len(self.layers_2)-1 and (i % self.skip) == 0:
        x = torch.cat([x, mid_init], dim=-1)
      x = layer(self.activation(x, inplace=True))

    out_size = self.out.out_features
    return self.out(self.activation(x, inplace=True)).reshape(batches + (out_size,))

#class NormalMLP(nn.Module):
#  "Two stage Skip Connected MLP with fourier encoding"
#  def __init__(
#    self,
#
#    num_layers = 3,
#    hidden_size=64,
#    ins=3,
#    intermediate=2,
#    out=3,
#    #code_size=64
#
#    skip=2,
#    # same frequencies are used for both inputs
#    freqs = [2**4, 2**5, 2**5, 2**5, 2**6, 2**6, 2**6, 2**7, 2**7, 2**8],
#    device="cuda",
#    activation = F.relu,
#  ):
#    super(NormalMLP, self).__init__()
#    self.ins = ins
#
#    n_f = len(freqs)
#    self.basis = create_fourier_basis([ins], freqs, n_f, device)
#
#    self.dim_1 = ins + 2 * n_f
#    self.skip = skip
#    skip_size = hidden_size + self.dim_1
#
#    hidden_layers_1 = [
#      nn.Linear(
#        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size,
#        hidden_size,
#      ) for i in range(num_layers)
#    ]
#    self.init =  nn.Linear(self.dim_1, hidden_size)
#    self.layers_1 = nn.ModuleList(hidden_layers_1)
#
#    self.inter = nn.Linear(hidden_size, intermediate)
#    self.from_inter =  nn.Linear(intermediate, hidden_size)
#    hidden_layers_2 = [
#      nn.Linear(
#        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size,
#        hidden_size,
#      ) for i in range(num_layers)
#    ]
#    self.layers_2 = nn.ModuleList(hidden_layers_2)
#    self.out =  nn.Linear(hidden_size, out)
#
#    self.activation = activation
#
#  def forward(self, i1):
#    batches = i1.shape[:-1]
#
#    init = fourier(i1.reshape(-1, self.ins), self.basis)
#    x = self.init(init)
#    for i, layer in enumerate(self.layers_1):
#      if i != len(self.layers_1)-1 and (i % self.skip) == 0:
#        x = torch.cat([x, init], dim=-1)
#      x = layer(self.activation(x, inplace=True))
#    x = self.inter(self.activation(x, inplace=True))
#    x = self.from_inter(self.activation(x, inplace=True))
#    for i, layer in enumerate(self.layers_2):
#      if i != len(self.layers_2)-1 and (i % self.skip) == 0:
#        x = torch.cat([x, init], dim=-1)
#      x = layer(self.activation(x, inplace=True))
#
#    out_size = self.out.out_features
#    return self.out(self.activation(x, inplace=True)).reshape(batches + (out_size,))

class AutoDecoder(nn.Module):
  def __init__(
    self,
    in_size=3,
    out=3,
    num_layers=4,
    w_in=True,
    code_size=64,
    freqs=[2**4, 2**4, 2**5, 2**5, 2**6, 2**6, 2**7, 2**7],
    hidden_size=64,
    skip=3,
    device="cuda",

    activation = F.leaky_relu,
  ):
    super(AutoDecoder, self).__init__()
    self.code = torch.rand(code_size, device=device, requires_grad=True)

    self.in_size = in_size
    self.w_in = w_in = in_size if w_in else 0

    n_p = len(freqs)
    self.basis_p = create_fourier_basis([w_in], freqs, n_p, device)

    self.dim_p = code_size + w_in + 2 * n_p
    self.skip = skip
    skip_size = hidden_size + self.dim_p

    # with skip
    hidden_layers = [
      nn.Linear(
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size,
        hidden_size,
      ) for i in range(num_layers)
    ]
    self.init =  nn.Linear(self.dim_p, hidden_size)
    self.layers = nn.ModuleList(hidden_layers)
    self.out = nn.Linear(hidden_size, out)

    self.activation = activation

  def forward(self, p):
    batches = p.shape[:-1]
    init = fourier(p.reshape(-1, self.in_size), self.basis_p)
    init = torch.cat([
      self.code.expand(init.shape[0], -1),
      init,
    ], dim=-1)
    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i != len(self.layers)-1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x))
    return self.out(self.activation(x)).reshape(*batches + (-1,))
  def latent_parameters(self): return [self.code]
  def randomize_code(self): self.code = torch.randn_like(self.code, requires_grad=True)
  def set_code(self, code):
    assert(self.code.shape == code.shape)
    self.code = code

class PartitionedAutoDecoder(nn.Module):
  def __init__(
    self,
    partition_fn=cartesian_indices_avgs,
    code_size=64,
    partition_size=8,
    in_size=3,
    out=3,
    num_layers=4,
    w_in=True,
    freqs=[2**4, 2**4, 2**5, 2**5, 2**6, 2**6, 2**7, 2**7],
    hidden_size=64,
    skip=3,
    device="cuda",

    activation = F.leaky_relu,
  ):
    super(PartitionedAutoDecoder, self).__init__()
    self.code_size = code_size
    self.partition_size = ps = partition_size
    # Decoding parameters
    self.code = nn.Parameter(
      torch.randn((ps * ps * ps, code_size), device=device, requires_grad=True)
    )
    self.partition_fn = partition_fn

    self.in_size = in_size
    self.w_in = w_in = in_size if w_in else 0

    n_p = len(freqs)
    self.basis_p = create_fourier_basis([w_in], freqs, n_p, device)

    self.dim_p = code_size + w_in + 2 * n_p
    self.skip = skip
    skip_size = hidden_size + self.dim_p

    # with skip
    hidden_layers = [
      nn.Linear(
        skip_size if (i % skip) == 0 and i != num_layers-1 else hidden_size, hidden_size,
      ) for i in range(num_layers)
    ]
    self.init =  nn.Linear(self.dim_p, hidden_size)
    self.layers = nn.ModuleList(hidden_layers)

    self.out = nn.Linear(hidden_size, out)
    self.activation = activation

  def forward(self, p):
    batches = p.shape[:-1]

    idxs, local = self.partition_fn(p.reshape(-1, self.in_size))

    x, y, z = [v.squeeze(-1) for v in idxs.split(1, dim=-1)]
    ps = self.partition_size

    idx = x + y * ps + z * ps * ps
    # a lil faster than indexing along each dimension?
    codes = self.code[idx]
    init = torch.cat([ codes, fourier(local, self.basis_p) ], dim=-1)
    x = self.init(init)
    for i, layer in enumerate(self.layers):
      if i != len(self.layers)-1 and (i % self.skip) == 0:
        x = torch.cat([x, init], dim=-1)
      x = layer(self.activation(x, inplace=True))
    return self.out(self.activation(x, inplace=True)).reshape(batches + (-1,))
  @torch.jit.ignore
  def latent_parameters(self): return [self.code]
  @torch.jit.ignore
  def non_latent_parameters(self):
    return [p for p in self.parameters() if p is not self.code]
  @torch.jit.ignore
  def randomize_code(self):
    self.code = nn.Parameter(torch.randn_like(self.code, requires_grad=True))
  # returns old code after being set
  @torch.jit.ignore
  def set_code(self, code):
    old = self.code.data
    assert(self.code.shape == code.shape)
    self.code = nn.Parameter(code)
    return old

class DensityEstimator(nn.Module):
  def __init__(
    self,
    in_size=2,
    dists=2<<4,
    device="cuda",
  ):
    super().__init__()
    self.centers = torch.zeros(dists, in_size, device=device, requires_grad=True)
    self.centers = nn.Parameter(self.centers)
    self.vars = torch.zeros((in_size * (in_size+1))//2, device=device, requires_grad=True)\
      .unsqueeze(0)\
      .repeat(dists, 1)\
      .detach()
    self.vars = nn.Parameter(self.vars)
    self.in_size = in_size
    self.weights = nn.Parameter(torch.zeros(dists, device=device, requires_grad=True))
  def forward(self, shape):
    a, d0, d1 = self.vars.split(1, dim=-1)
    z = torch.zeros_like(a)
    scale_tril = torch.cat([
      d0.exp(), z,
      a, d1.exp(),
    ], dim=-1).reshape(-1, self.in_size, self.in_size)
    dist = D.MultivariateNormal(self.centers, scale_tril=scale_tril)
    out = dist.rsample(shape)
    k = F.softmax(self.weights, dim=-1)
    out = out.permute(4, 0, 1, 2, 3, 5)
    val = out * k[:, None, None, None, None, None].expand_as(out)
    val = val.sum(dim=0)
    pdf = (dist.log_prob(val).exp() * k[None, None, None, :]).sum(dim=-1)
    assert((pdf <= 1.).all())
    assert((pdf >= 0.).all())
    return val, pdf
  def pdf(self, val):
    a, d0, d1 = self.vars.split(1, dim=-1)
    z = torch.zeros_like(a)
    scale_tril = torch.cat([
      d0.exp(), z,
      a, d1.exp(),
    ], dim=-1).reshape(-1, self.in_size, self.in_size)
    dist = D.MultivariateNormal(self.centers, scale_tril=scale_tril)
    k = F.softmax(self.weights, dim=-1)
    pdf_indiv = dist.log_prob(val.unsqueeze(-2)).exp()
    pdf = (pdf_indiv * k.expand_as(pdf_indiv)).sum(dim=-1,keepdim=True)
    return pdf


# Given an image returns a latent code for it
class Embedder(nn.Module):
  def __init__(
    self,
  ):
    super().__init__()
  def forward(self, img, word):
    # TODO some number of convolutional layers then MLP to return feature vector
    raise NotImplementedError()

# simple gan discriminator
class Discriminator(nn.Module):
  def __init__(
    self,
    num_features = 64,
    num_channel = 3,
    device="cuda",
  ):
    super().__init__()
    self.main = nn.Sequential(
      # input is (nc) x 64 x 64
      nn.Conv2d(num_channel, num_features, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 32 x 32
      nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(num_features * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 16 x 16
      nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(num_features * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(num_features * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False),
    )
  def forward(self, x):
    assert(len(x.shape) == 4)
    _, C, W, H = x.shape
    #assert(C == 3 and W == 64 and H == 64)
    return self.main(x).reshape(-1)
