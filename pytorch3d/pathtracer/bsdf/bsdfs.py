from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch3d.pathtracer.warps import (
  square_to_cos_hemisphere, square_to_cos_hemisphere_pdf,
  NeuralWarp, MipMap,
)
from pytorch3d.pathtracer.utils import (
  pos_weak_sigmoid, cartesian_to_log_polar, param_rusin,
  param_rusin2, fwidth, dir_to_uv, weak_sigmoid,
)
from pytorch3d.pathtracer.neural_blocks import ( SkipConnMLP, DensityEstimator )
from itertools import chain
import pytorch3d.pathtracer as pt
from ..interaction import ( partial_frame, to_local )

# A sample of a light bounce from a BSDF
@dataclass
class BSDFSample:
  pdf: torch.Tensor = 0
  wo: torch.Tensor = 0
  eta: torch.Tensor = 1
  # Combines two mutually exclusive BSDF samples
  def combine(self, other, mask_self, mask_other):
    pdf = torch.where(mask_self, self.pdf,
      torch.where(mask_other, other.pdf, torch.zeros_like(other.pdf)))

    wo = torch.where(mask_self, self.wo,
      torch.where(mask_other, other.wo, torch.zeros_like(other.wo)))

    return BSDFSample(pdf=pdf, wo=wo, eta = self.eta)
  # Empty BSDF sample
  @classmethod
  def zeros_like(cls, like):
    return cls(
      pdf = torch.zeros(like.shape[:-1], device=like.device),
      wo = torch.zeros_like(like),
      eta = 1, # :)
    )
  # Composes together a weighted sample of BSDFs (like a gather almost)
  @staticmethod
  def compose(samples: ["BSDFSample"], k, selections):
    pdfs = torch.stack([s.pdf for s in samples], dim=-1)\
      .reshape(-1, k.shape[-1])
    pdf = pdfs[range(pdfs.shape[0]), selections]
    # Have to multiply pdf by k since it's joint likelihood of selecting item
    pdf = pdf * k.reshape(-1, k.shape[-1])[range(pdf.shape[0]), selections]
    pdf = pdf.reshape_as(samples[0].pdf)
    wos = torch.stack([s.wo for s in samples], dim=-1)\
      .reshape(-1, 3, k.shape[-1])
    wo = wos[range(wos.shape[0]), :, selections].reshape_as(samples[0].wo)
    return BSDFSample(
      pdf = pdf,
      wo = F.normalize(wo, dim=-1),
      # FIXME well it's not currently used...
      eta = samples[0].eta,
    )


# General interface for a BSDF
class BSDF(nn.Module):
  def __init__(self):
    super().__init__()
  def sample(self, it, sampler, active=True): raise NotImplementedError()
  def eval_and_pdf(self, it, wo, active=True): raise NotImplementedError()
  def joint_eval_pdf(self, it, wo, active=True):
    spectrum, pdf = self.eval_and_pdf(it, wo, active)
    return torch.cat([ spectrum, pdf.reshape(spectrum.shape[:-1] + (1,)) ], dim=-1)
  def eval(self, it, wo, active=True): return self.eval_and_pdf(it, wo, active)[0]
  def pdf(self, it, wo, active=True): return self.eval_and_pdf(it, wo, active)[1]

def identity(x): return x
def identity_div_pi(x): return x/math.pi

# Diffuse BSDF with additional preprocessing (postprocessing?) functionality.
class Diffuse(BSDF):
  def __init__(self, reflectance=[0.25, 0.2, 0.7], preprocess=identity_div_pi, device="cuda"):
    super().__init__()
    if type(reflectance) == list:
      self.reflectance = torch.tensor(reflectance, device=device, requires_grad=True)
    else:
      self.reflectance = reflectance
    self.preproc = preprocess
  def parameters(self): return [self.reflectance]
  def random(self):
    self.reflectance = torch.rand_like(self.reflectance, requires_grad=True)
    return self
  def sample(self, it, sampler, active=True):
    cos_theta_i = it.wi[..., 2]

    bs = BSDFSample.zeros_like(it.p)

    active = (cos_theta_i > 0) & active
    if not active.any(): return bs, torch.zeros_like(it.p)

    bs.wo = square_to_cos_hemisphere(sampler.sample(it.shape()[:-1] + (2,), device=it.device()))
    bs.wo = F.normalize(bs.wo, dim=-1)
    bs.pdf = square_to_cos_hemisphere_pdf(bs.wo)
    bs.eta = 1.0
    bs.sampled_component = 0
    # cast spectrum to same shape as interaction
    spectrum = self.preproc(self.reflectance).expand(*it.shape()).clone()
    #spectrum[(~active) | (bs.pdf <= 0), :] = 0
    return bs, spectrum

  def eval_and_pdf(self, it, wo, active=True):
    cos_theta_i = it.wi[..., 2]
    cos_theta_o = wo[..., 2]
    #active = (cos_theta_i > 0) & (cos_theta_o > 0) & active

    spectrum = self.preproc(cos_theta_o.unsqueeze(-1) * self.reflectance)
    #spectrum[~active] = 0
    pdf = square_to_cos_hemisphere_pdf(wo)
    #pdf[~active] = 0

    return spectrum, pdf


# Reflection vector
@torch.jit.script
def reflect(n, v): return 2 * (n * v).sum(keepdim=True, dim=-1) * n - v

# Reflection vector in local frame
@torch.jit.script
def local_reflect(v):
  x, y, z = v.split(1, dim=-1)
  return torch.cat([-x, -y, z], dim=-1)

# A Phong BSDF with a few parameters and pre/post processing.
class Phong(BSDF):
  def __init__(
    self,
    diffuse=[0.6, 0.5, 0.7],
    specular=[0.8, 0.8, 0.8],
    min_spec=1,
    device="cuda",
  ):
    super().__init__()
    if type(diffuse) == list:
      self.diffuse = torch.tensor(diffuse, device=device, requires_grad=True)
    else: self.diffuse = diffuse
    if type(specular) == list:
      self.specular = torch.tensor(specular, device=device, requires_grad=True)
    else: self.specular = specular
    self.shine = torch.tensor(40., dtype=torch.float, device=device, requires_grad=True)
    self.min_spec = min_spec
  def parameters(self): return [self.specular, self.diffuse, self.shine]
  def random(self):
    self.shine = torch.rand_like(self.shine, requires_grad=True)
    self.specular = torch.rand_like(self.specular, requires_grad=True)
    self.diffuse = torch.rand_like(self.diffuse, requires_grad=True)
    return self
  def sample(self, it, sampler, active=True):
    cos_theta_i = it.wi[..., 2]

    bs = BSDFSample.zeros_like(it.p)

    active = (cos_theta_i > 0) & active
    if not active.any(): return bs, torch.zeros_like(it.p)

    bs.wo = square_to_cos_hemisphere(sampler.sample(it.shape()[:-1] + (2,), device=it.device()))
    bs.pdf = square_to_cos_hemisphere_pdf(bs.wo)
    bs.eta = 1.0
    bs.sampled_component = 0
    # cast spectrum to same shape as interaction
    cos_theta_o = bs.wo[..., 2]
    active = (cos_theta_o > 0) & active
    R = reflect(it.frame[..., 2], it.wi)
    spectral = (R * bs.wo).sum(dim=-1).clamp(min=1e-20).pow(self.min_spec + self.shine.exp())
    spectrum = cos_theta_i.unsqueeze(-1) * self.diffuse/math.pi + \
               spectral.unsqueeze(-1) * self.specular/math.pi
    spectrum[(~active) | (bs.pdf <= 0), :] = 0
    return bs, spectrum
  def eval_and_pdf(self, it, wo, active=True):
    cos_theta_i = it.wi[..., 2]
    cos_theta_o = wo[..., 2]
    # active = (cos_theta_i > 0) & (cos_theta_o > 0) & active
    R = reflect(it.frame[..., 2], it.wi)
    spectral = (R * wo).sum(dim=-1).clamp(min=1e-20).pow(self.min_spec + self.shine.exp())
    spectrum = cos_theta_i.unsqueeze(-1) * self.diffuse/math.pi + \
               spectral.unsqueeze(-1) * self.specular/math.pi
    # just a guess of the PDF since it's not physically based
    pdf = square_to_cos_hemisphere_pdf(wo)

    #spectrum[~active] = 0
    #pdf[~active] = 0
    return spectrum, pdf

# fresnel and fresnel_diff_refl taken from Mitsuba
# https://github.com/mitsuba-renderer/mitsuba2/blob/master/include/mitsuba/render/fresnel.h
def fresnel(cos_t, eta: float):
  def fnma(x, y, z):  return -x * y + z
  def fma(x, y, z): return x * y + z
  out_mask = (cos_t >= 0)
  inv_eta = 1/eta
  eta_it = torch.where(out_mask, eta, inv_eta)
  eta_ti = torch.where(out_mask, inv_eta, eta)
  cos_tt_sqr = fnma(fnma(cos_t, cos_t, 1), eta_ti * eta_ti, 1)
  cos_t_abs = cos_t.abs()
  cos_tt_abs = cos_tt_sqr.clamp(min=1e-10).sqrt()

  idx_match = (eta == 1)
  special_case = (cos_t_abs == 0) | idx_match


  a_s = fnma(eta_it, cos_tt_abs, cos_t_abs)/\
         fma(eta_it, cos_tt_abs, cos_t_abs)
  a_p = fnma(eta_it, cos_t_abs, cos_tt_abs)/\
         fma(eta_it, cos_t_abs, cos_tt_abs)

  r = 0.5 * (a_s.square() + a_p.square())
  r[special_case] = 0 if idx_match else 1

  cos_tt = cos_tt_abs * -cos_t.sign()

  return r, cos_tt, eta_it, eta_ti

def fresnel_diff_refl(eta):
  if eta < 1:
     return -1.4399 * (eta * eta) \
            + 0.7099 * eta \
            + 0.6681 \
            + 0.0636 / eta
  inv_eta = 1/eta
  inv_eta_2 = inv_eta   * inv_eta
  inv_eta_3 = inv_eta_2 * inv_eta
  inv_eta_4 = inv_eta_3 * inv_eta
  inv_eta_5 = inv_eta_4 * inv_eta
  return 0.919317 - 3.4793 * inv_eta \
         + 6.75335 * inv_eta_2 \
         - 7.80989 * inv_eta_3 \
         + 4.98554 * inv_eta_4 \
         - 1.36881 * inv_eta_5

# A BSDF for representing plastic as per Mitsuba.
class Plastic(BSDF):
  def __init__(
    self,
    diffuse=[0.5, 0.5, 0.5],
    specular=[1.,1.,1.],
    int_ior:float=1.49, ext_ior:float=1.000277,
    device="cuda",
  ):
    if type(diffuse) == list:
      self.diffuse = torch.tensor(diffuse, device=device, requires_grad=True)
    else: self.diffuse = diffuse
    if type(specular) == list:
      self.specular = torch.tensor(specular, device=device, requires_grad=True)
    else: self.specular = specular
    assert(int_ior > 0)
    assert(ext_ior > 0)
    self.eta = int_ior/ext_ior

    self.inv_eta_2 = 1/(self.eta * self.eta)

    self.fdr_int = fresnel_diff_refl(1/self.eta)
    self.fdr_ext = fresnel_diff_refl(self.eta)


  def spec_sample_weight(self):
    d = self.diffuse.mean()
    s = self.specular.mean()
    return s/(d+s)
  def parameters(self): return [self.diffuse, self.specular]
  def random(self):
    self.specular = torch.rand_like(self.specular, requires_grad=True)
    self.diffuse = torch.rand_like(self.diffuse, requires_grad=True)
    return self
  def eval_and_pdf(self, it, wo, active=True):
    cos_theta_i = it.wi[..., 2]
    cos_theta_o = wo[..., 2]
    active = (cos_theta_i > 0) & (cos_theta_o > 0) & active
    f_i = fresnel(cos_theta_i, self.eta)[0]
    f_o = fresnel(cos_theta_o, self.eta)[0]
    pdf = square_to_cos_hemisphere_pdf(wo)
    spectrum = (self.diffuse.expand_as(it.p)/(1 - self.fdr_int)) \
      *  self.inv_eta_2 * (pdf * (1 - f_i) * (1 - f_o)).unsqueeze(-1)

    # DeltaReflection

    ssw = self.spec_sample_weight()
    prob_specular = ssw * f_i
    prob_diffuse = (1 - f_i) * (1-ssw)
    prob_diffuse = prob_diffuse/(prob_specular + prob_diffuse)
    pdf = pdf * prob_diffuse

    #spectrum[~active] = 0
    #pdf[~active] = 0
    return spectrum, pdf
  def sample(self, it, sampler, active=True):
    bs = BSDFSample.zeros_like(it.p)
    spectrum = torch.zeros_like(it.p)

    cos_theta_i = it.wi[..., 2]
    active = (cos_theta_i > 0) & active
    f_i = fresnel(cos_theta_i, self.eta)[0]
    spec_sample_weight = self.spec_sample_weight()
    p_spec = f_i * spec_sample_weight
    p_diff = (1 - f_i) * (1 - spec_sample_weight)
    p_spec = (p_spec)/(p_spec + p_diff)
    p_diff = 1 - p_spec
    sample_spec = active & (sampler.sample(p_spec.shape) < p_spec)
    # sample_diff = active & (~sample_spec)
    bs.wo = torch.where(
      sample_spec.unsqueeze(-1),
      reflect(it.frame[..., 2], it.wi),
      square_to_cos_hemisphere(sampler.sample(it.shape()[:-1] + (2,), device=it.device())),
    )
    bs.pdf = torch.where(
      sample_spec,
      p_spec,
      p_diff * square_to_cos_hemisphere_pdf(bs.wo),
    ).clamp(min=1e-10)
    f_o = fresnel(bs.wo[..., 2], self.eta)[0]
    spectrum = torch.where(
      sample_spec.unsqueeze(-1),
      self.specular * (f_i/bs.pdf).unsqueeze(-1),
      self.diffuse.expand_as(it.p) / (1- self.fdr_int) \
      *  bs.pdf.unsqueeze(-1) * self.inv_eta_2 *\
      (1 - f_i.unsqueeze(-1)) * (1 - f_o.unsqueeze(-1))
    )

    return bs, spectrum

@torch.jit.script
def fresnel_conductor(cos_t, eta_r: float, eta_i: float):
  ct2 = cos_t * cos_t
  st2 = (1 - ct2).clamp(min=1e-10)
  st4 = st2 * st2
  tmp = eta_r * eta_r - eta_i * eta_i - st2
  a_2_pb_2 = (tmp*tmp + 4 * eta_i * eta_i * eta_r * eta_r).clamp(min=1e-10).sqrt()
  a = (0.5 * (a_2_pb_2 + tmp)).clamp(min=1e-10).sqrt()
  t1 = a_2_pb_2 + ct2
  t2 = 2 * cos_t * a
  r_s = (t1 - t2)/(t1 + t2)
  t3 = a_2_pb_2 * ct2 + st4
  t4 = t2 * st2
  r_p = r_s * (t3 - t4) / (t3 + t4)
  return 0.5 * (r_s + r_p)

# A BSDF for representing an entirely reflective conductor.
# Not thoroughly tested but should generally work.
class Conductor(BSDF):
  def __init__(
    self,
    specular=[1.,1.,1.],
    eta:float=1.3,
    k:float=1,
    device="cuda",
    activation = torch.sigmoid,
  ):
    super().__init__()
    self.eta = torch.tensor(eta, requires_grad=True, dtype=torch.float)
    self.k = torch.tensor(k, requires_grad=True, dtype=torch.float)
    if type(specular) == list:
      self.specular = torch.tensor(specular, device=device, requires_grad=True)
    else: self.specular = specular
    self.act = activation
  def random(self):
    self.specular = torch.rand_like(self.specular, requires_grad=True)
    return self
  def eval_and_pdf(self, it, wo, active=True):
    spectrum = torch.zeros_like(it.p)
    pdf = torch.zeros(it.p.shape[:-1], dtype=torch.float, device=it.p.device)
    #active = (it.wi[..., 2] > 0) & (wo[..., 2] > 0) & active

    refl = local_reflect(it.wi)
    thresh = (refl * wo).sum(dim=-1, keepdim=True) > 0.94
    fresnel = fresnel_conductor(it.wi[..., 2], F.softplus(self.eta), 0.0).reshape_as(thresh)
    spectrum = torch.where(
      thresh,
      fresnel * self.act(self.specular),
      torch.zeros_like(spectrum),
    )
    pdf = torch.where(
      thresh.reshape_as(pdf),
      torch.ones_like(pdf),
      torch.zeros_like(pdf),
    )

    spectrum = torch.where(
      active.unsqueeze(-1),
      spectrum, torch.zeros_like(spectrum),
    )

    return spectrum, pdf

  def parameters(self): return [self.eta, self.k, self.specular]
  def sample(self, it, sampler, active=True):
    cos_theta_i = it.wi[..., 2]
    active = (cos_theta_i > 0) & active
    bs = BSDFSample.zeros_like(it.p)
    spectrum = torch.zeros_like(it.p)
    bs.wo  = reflect(it.wi)
    bs.eta = 1
    bs.pdf = torch.ones_like(active)

    spectrum[active] = self.specular * fresnel_conductor(cos_theta_i, self.eta, self.k)[active]
    return bs, spectrum

# inverts a direction along the z-axis
def invert_z(xyz) -> torch.Tensor:
  x, y, z = xyz.split(1, dim=-1)
  return torch.cat([x, y, -z], dim=-1)

# A 2-Sided BSDF, which by default makes both sides one BSDF.
class Bidirectional(BSDF):
  def __init__(self, front, back=None):
    super().__init__()
    self.front = front
    if back is None: back = front
    self.back = back

  def sample(self, it, sampler, active=True):
    cos_theta_i = it.wi[..., 2]
    front = (cos_theta_i > 0) & active
    back = (cos_theta_i < 0) & active
    front_bs, front_spectrum = self.front.sample(it, sampler, front)

    # perform back-side sampling
    original_wi = it.wi
    it.wi = invert_z(it.wi)
    back_bs, back_spectrum = self.back.sample(it, sampler, back)
    back_bs.wo = invert_z(back_bs.wo)
    it.wi = original_wi

    spectrum = torch.where(front, front_spectrum,
      torch.where(back, back_spectrum, torch.zeros_like(back_spectrum)))

    return front_bs.combine(back_bs), spectrum

  def eval_and_pdf(self, it, wo, active=True):
    cos_theta_i = it.wi[..., 2]

    front = (cos_theta_i > 0) & active
    back = (cos_theta_i < 0) & active

    front_eval, front_pdf = self.front.eval_and_pdf(it, wo, front)

    og_wi = it.wi
    it.wi = invert_z(og_wi)
    back_eval, back_pdf = self.back.eval_and_pdf(it, invert_z(wo), back)
    it.wi = og_wi

    spectrum = torch.where(front.unsqueeze(-1), front_eval,
      torch.where(back.unsqueeze(-1), back_eval, torch.zeros_like(back_eval)))

    pdf = torch.where(front, front_pdf,
      torch.where(back, back_pdf, torch.zeros_like(back_pdf)))

    return spectrum, pdf

# Composes a bunch of BSDFs together using some static weighting (not spatially varying)
class Compose(BSDF):
  def __init__(self, bsdfs: [BSDF], device="cuda"):
    # have to keep it as a list but I wish I could dispatch to all of them simultaneously
    # aghhhh pythonnnnn
    self.bsdfs = bsdfs
    self.weights = torch.rand(len(bsdfs), device=device) + 0.5
  def sample(self, it, sampler, active=True):
    raise NotImplementedError()
  def eval_and_pdf(self, it, wo, active=True):
    spec_pdf = self.normalized_weights() * torch.stack([
      bsdf.joint_eval_pdf(it, wo, active)
      for bsdf in self.bsdfs
    ], dim=-1)
    spectrum, pdf = spec_pdf.sum(dim=-1).split([3, 1], dim=-1)
    return spectrum, pdf.squeeze(-1)
  def normalized_weights(self):
    return F.softmax(self.weights, dim=-1)
  def parameters(self):
    return chain(
      *[bsdf.parameters() for bsdf in self.bsdfs],
      [self.weights],
    )
  def own_parameters(self): return [self.weights]

# A spatially-varying BSDF which is determined by some function f(xyz) -> [# BSDFS].
# By default it is a learned composition, but it can be a normal function as well.
class ComposeSpatialVarying(BSDF):
  def __init__(self, bsdfs: [BSDF], spatial_varying_fn= None, device="cuda"):
    super().__init__()
    self.bsdfs = bsdfs
    if spatial_varying_fn is None:
      self.sp_var_fn = SkipConnMLP(
        num_layers=16,
        hidden_size=256,
        freqs=128,
        sigma=2<<6,
        in_size=3, out=len(bsdfs),
        device=device,

        xavier_init=True,
      ).to(device)
    else:
      self.sp_var_fn = spatial_varying_fn
    self.preprocess = identity
  def sample(self, it, sampler, active=True):
    bsdf_samples, spectrums = list(zip(
      *[bsdf.sample(it, sampler, active) for bsdf in self.bsdfs]
    ))

    k = self.normalized_weights(it.p, it)
    selections = torch.multinomial(k.reshape(-1, len(self.bsdfs)), num_samples=1).squeeze(-1)
    spectrums = torch.stack(spectrums, dim=-1).reshape(-1, 3, len(self.bsdfs))
    spectrum = spectrums[range(spectrums.shape[0]), :, selections]\
      .reshape_as(it.p)
    bs = BSDFSample.compose(bsdf_samples, k, selections)
    # how does one sample from a linear combination of BSDFs?
    # This is just an approximation by sampling from one of them
    return bs, spectrum

  def eval_and_pdf(self, it, wo, active=True):
    k = self.normalized_weights(it.p, it)
    spec_pdf = torch.stack([
      bsdf.joint_eval_pdf(it, wo, active) for bsdf in self.bsdfs
    ], dim=-1)
    setattr(it, 'normalized_weights', k)
    spec_pdf = torch.where(
      active[..., None, None],
      spec_pdf * k.unsqueeze(-2),
      torch.zeros_like(spec_pdf),
    )
    spectrum, pdf = spec_pdf.sum(dim=-1).split([3, 1], dim=-1)

    return spectrum, pdf.squeeze(-1)
  def normalized_weights(self, p, it):
    out_shape = p.shape[:-1] + (len(self.bsdfs),)
    weights = self.sp_var_fn(self.preprocess(p))
    weights = weights.reshape(out_shape)
    setattr(it, 'nonnormalized_weights', weights)
    #return F.softmax(weights, dim=-1)
    # Softmax seems to have some issues with local minima, so below also works
    return weights.sigmoid()
  def parameters(self): return chain(self.own_parameters(), self.child_parameters())
  def own_parameters(self): return self.sp_var_fn.parameters()
  def child_parameters(self): return chain(*[bsdf.parameters() for bsdf in self.bsdfs])

# Hard classifier of BSDFs (used during experimentation)
class SelectBSDF(BSDF):
  def __init__(self, selector, bsdfs, device="cuda"):
    super().__init__()
    self.selector = selector
    self.bsdfs = bsdfs
  def select(self, p):
    return self.selector(p)
  def parameters(self):
    return chain(*[bsdf.parameters() for bsdf in self.bsdfs])
    return chain(self.attenuation.parameters(), self.color.parameters(), self.dist.parameters())
  def sample(self, it, sampler, active=True):
    raise NotImplementedError()
    return bs, spectrum

  def eval_and_pdf(self, it, wo, active=True):
    spec_pdf = torch.stack([
      bsdf.joint_eval_pdf(it, wo, active) for bsdf in self.bsdfs
    ], dim=-1)
    i = self.select(it.p)
    flat_spec_pdf = spec_pdf.reshape(-1, 4, len(self.bsdfs))
    spec_pdf = flat_spec_pdf[range(flat_spec_pdf.shape[0]), :, i]\
      .reshape(spec_pdf.shape[:-1])
    spectrum, pdf = spec_pdf.split([3, 1], dim=-1)
    return spectrum, pdf.squeeze(-1)


# One big MLP for both spatially-varying and coloration (used while developing)
class GlobalNeuralBSDF(BSDF):
  def __init__(self, device="cuda"):
    super().__init__()
    self.attenuation = SkipConnMLP(
      in_size=3, out=1,
      num_layers=3, hidden_size=64,
      activation=F.relu,
      device=device,
    ).to(device)
    self.color = SkipConnMLP(
      in_size=3, out=3,
      num_layers=3, hidden_size=64,
      activation=F.relu,
    ).to(device)
    self.dist = NeuralWarp(device=device).to(device)
  def parameters(self):
    return chain(self.attenuation.parameters(), self.color.parameters(), self.dist.parameters())
  def random(self): return self
  def sample(self, it, sampler, active=True):
    bs = BSDFSample.zeros_like(it.p)

    direc, pdf = self.dist(it.p.shape[:-1])
    bs.wo = F.normalize(direc, eps=1e-7, dim=-1)
    bs.pdf = pdf.unsqueeze(-1)
    bs.eta = 1.0
    bs.sampled_component = 0
    # cast spectrum to same shape as interaction
    attenuation = (1+self.attenuation(param_rusin(it.n, it.wi, bs.wo)))/2
    spectrum = attenuation * ((1 + self.color(it.p))/2)
    w = (0.5 * fwidth(spectrum))
    spectrum = spectrum * w.sin()/w.clamp(min=1e-7)

    spectrum[(~active) | (bs.pdf <= 0), :] = 0
    return bs, spectrum
  def eval_and_pdf(self, it, wo, active=True):
    attenuation = self.attenuation(param_rusin(it.n, it.wi, wo))
    spectrum = attenuation * ((1 + self.color(it.p))/2)
    w = (0.5 * fwidth(spectrum))
    spectrum = spectrum * w.sin()/w.clamp(min=1e-7)
    pdf = self.dist.pdf(dir_to_uv(wo)).unsqueeze(-1)
    return spectrum, pdf


# A single component BSDF which is a just neural net f(rusin) -> RGB
class NeuralBSDF(BSDF):
  def __init__(self, activation=torch.sigmoid, device="cuda"):
    super().__init__()
    self.mlp = SkipConnMLP(
      in_size=3, out=3,
      num_layers=6, hidden_size=96,
      freqs=64,
      device=device,
    ).to(device)
    self.act = activation
  def parameters(self): return chain(self.mlp.parameters())
  def random(self): return self
  def sample(self, it, sampler, active=True):
    cos_theta_i = it.wi[..., 2]
    bs = BSDFSample.zeros_like(it.p)
    bs.wo = square_to_cos_hemisphere(sampler.sample(it.shape()[:-1] + (2,), device=it.device()))
    bs.wo = F.normalize(bs.wo, dim=-1)
    bs.pdf = square_to_cos_hemisphere_pdf(bs.wo)
    bs.eta = 1.0
    spectrum = self.act(self.mlp(param_rusin2(it.wi, bs.wo)))
    return bs, spectrum
  def eval_and_pdf(self, it, wo, active=True):
    spectrum = self.act(self.mlp(param_rusin2(it.wi, wo)))
    pdf = torch.ones(spectrum.shape[:-1], device=spectrum.device)
    return spectrum, pdf
  # Zeros out this MLP so that it doesn't return any colors. Can be useful when analyzing their
  # outputs.
  def zero(self):
    class Zero(nn.Module):
      def __init__(self): super().__init__()
      def forward(self, x): return torch.zeros_like(x)
    self.mlp = Zero()

