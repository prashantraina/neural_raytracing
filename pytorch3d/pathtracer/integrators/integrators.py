import torch
import torch.nn as nn
import torch.nn.functional as F

from ..scene import (
  sample_emitter_dir_w_isect,
  sample_emitter_dir_wo_isect,
  sample_emitter_dir_w_learned_occ,
)
from ..neural_blocks import ( SkipConnMLP )
import pytorch3d.pathtracer as pt

# General integration interface
class Integrator(nn.Module):
  def __init__(self, max_depth=2, russian_roulette_depth=5, sampler=None, lights=None):
    super().__init__()
    self.max_depth=max_depth
    self.rr_depth = russian_roulette_depth
    self.sampler = sampler
    self.lights = lights
  def dims(self): raise NotImplementedError()
  def sample(self, shapes, rays, bsdf, **kwargs): raise NotImplementedError()

# Returns the normals of a surface, useful when debugging
class Debug(Integrator):
  def dims(self): return 3
  def sample(self, shapes, rays, bsdf, **kwargs):
    si, active = shapes.intersect(rays)
    #result = si.n
    result = torch.where(
      active.unsqueeze(-1),
      (si.n + 1)/2,
      torch.tensor(0., device=active.device, dtype=torch.float)
    )
    return result, active, si

# Indicator function of intersection
class Silhouette(Integrator):
  def dims(self): return 1
  def sample(self, shapes, rays, bsdf, **kwargs):
    si, active = shapes.intersect(rays)
    return 1-active.unsqueeze(-1).float(), active, si

# Adds a mask of where shapes were hit based on some sub integrator.
class Mask(Integrator):
  def dims(self): return self.sub_integrator.dims() + 1
  def __init__(self, sub_integrator, **kwargs):
    super().__init__(**kwargs)
    self.sub_integrator = sub_integrator
  def sample(self, density_field, rays, bsdf, **kwargs):
    result, active, si = self.sub_integrator.sample(density_field, rays, bsdf, **kwargs)
    mask = torch.where(active, 1., 0.)
    result = torch.cat([ result, mask.unsqueeze(-1) ], dim=-1)
    return result, torch.ones_like(active), si

# Returns the depths along a given ray, altho it might be a bit buggy?
class Depth(Integrator):
  def __init__(self, scale=False, empty_val=-1, **kwargs):
    super().__init__(**kwargs)
    self.empty_val = empty_val
    self.scale = scale
  def dims(self): return 1
  def sample(self, shapes, rays, bsdf, **kwargs):
    it, active = shapes.intersect(rays)
    results = torch.where(active, it.t, torch.full_like(it.t, self.empty_val))
    if self.scale: results[results != 0] /= results[results != 0].max()
    return results.unsqueeze(-1), active, it

# Special integrator for SDFs which is defined over the whole image space
class LevelSurfaces(Integrator):
  def dims(self): return 1
  def sample(self, shapes, rays, bsdf, **kwargs):
    min_sdfs = shapes.level_surfaces(rays)
    #min sdfs in range [0, inf)
    monochrome = torch.exp(-min_sdfs.clamp(min=1e-10))
    return monochrome.unsqueeze(-1), torch.tensor(True, device=rays.device), None

# Renders the weight map of a spatially varying BSDF.
class BasisBRDF(Integrator):
  def __init__(self, multi_basis_bsdf):
    super().__init__()
    self.bsdf = multi_basis_bsdf
  def dims(self): return len(self.bsdf.bsdfs)
  def sample(self, shapes, rays, bsdf, **kwargs):
    device = rays.device
    results = torch.zeros(*rays.shape[:-1], self.dims(), device=device)
    it, active = shapes.intersect(rays)
    if not active.any(): return results, active, it
    results[active] = self.bsdf.normalized_weights(it.p[active], it)
    return results, active, it

# Renders light direction and intensity on surface
class Illumination(Integrator):
  def __init__(self):
    super().__init__()
  def dims(self): return 3
  def sample(self, shapes, rays, lights, sampler, **kwargs):
    device = rays.device
    sample_emitter = kwargs.get("sample_emitter_fn", sample_emitter_dir_wo_isect)
    it, active = shapes.intersect(rays)

    ds, emitter_val = sample_emitter(
      it, shapes, lights=lights, sampler=sampler, active=active,
    )

    results = torch.where(
      active.unsqueeze(-1),
      (F.normalize(it.to_local(ds.d), dim=-1)+1)/2,
      torch.zeros_like(ds.d),
    )
    return (1+results)/2, active, it

# Renders light direction and intensity on surface
class Luminance(Integrator):
  def __init__(self):
    super().__init__()
  def dims(self): return 3
  def sample(self, shapes, rays, lights, sampler, **kwargs):
    device = rays.device
    sample_emitter = kwargs.get("sample_emitter_fn", sample_emitter_dir_wo_isect)
    it, active = shapes.intersect(rays)

    ds, emitter_val = sample_emitter(
      it, shapes, lights=lights, sampler=sampler, active=active,
    )

    def lum(rgb):
      r,g,b = rgb.split(1, dim=-1)
      return 0.2126 * r + 0.7152 * 0.0722 * b

    results = torch.where(
      active.unsqueeze(-1),
      lum(emitter_val).expand_as(ds.d),
      torch.zeros_like(ds.d),
    )
    return results, active, it

# Direct lighting integrator
class Direct(Integrator):
  DEFAULT_EMITTER_SAMPLES = 1
  DEFAULT_BSDF_SAMPLES = 0

  def dims(self): return 3
  def __init__(
    self,
    emitter_samples = DEFAULT_EMITTER_SAMPLES,
    bsdf_samples = DEFAULT_BSDF_SAMPLES,
    training=True,
    **kwargs,
  ):
    self.emitter_samples = emitter_samples
    self.bsdf_samples = bsdf_samples
    self.training = training
    super().__init__(**kwargs)

  def sample(self, shapes, rays, bsdf, **kwargs):
    device = rays.device

    sampler = kwargs.get("sampler", self.sampler)
    lights = kwargs.get("lights", self.lights)
    w_isect = kwargs.get("w_isect")
    sample_emitter = sample_emitter_dir_wo_isect
    if w_isect is True: sample_emitter = sample_emitter_dir_w_isect
    if type(w_isect) is SkipConnMLP:
      sample_emitter=lambda it,s,lights,sampler,active: \
        sample_emitter_dir_w_learned_occ(it,s,lights,sampler,w_isect, active)

    result = torch.zeros(*rays.shape[:-1], 3, device=device)

    it, active = shapes.intersect(rays, primary=self.training)
    if not active.any(): return result, active, it

    for i in range(self.emitter_samples):
      ds, emitter_val = sample_emitter(
        it, shapes, lights=lights, sampler=sampler, active=active,
      )
      active_emitted = active & (ds.pdf > 0)
      wo = it.to_local(ds.d)
      bsdf_val, bsdf_pdf = bsdf.eval_and_pdf(it, wo, active=active_emitted)
      bsdf_pdf = bsdf_pdf.reshape_as(active_emitted)
      #active_emitted = active_emitted & (bsdf_pdf > 0)

      mis = torch.ones_like(bsdf_pdf, device=device)
      #mis[~ds.delta] = mis_weight(ds.pdf, bsdf_pdf)

      val = mis[active_emitted].unsqueeze(-1) * \
        bsdf_val[active_emitted] * emitter_val[active_emitted]
      val = val / self.emitter_samples
      result[active_emitted] = result[active_emitted] + val
    # TODO fill out below
    for i in range(self.bsdf_samples):
      bs, bsdf_val = bsdf.sample(it, sampler=sampler, active=active)
      rays = it.spawn_rays(it.from_local(bs.wo))
      bsdf_it, snd_active = lights.intersect(rays)
      bsdf_active = active & snd_active
      if bsdf_active.any():
        emit_val, emit_pdf = emitter.eval_pdf(it, bsdf_it)
        raise NotImplementedError()
        result = torch.where(
          bsdf_active,
          # TODO clean up the line below
          result + bsdf_val * emitter_val * emitter_pdf,
          result,
        )

    return result, active, it

class NeuralApprox(Integrator):
  def dims(self): return 3
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    from pytorch3d.pathtracer.neural_blocks import TwoStageMLP
    device = kwargs.get("device", "cuda")
    self.mlp = TwoStageMLP(device=device).to(device)
  def parameters(self): return self.mlp.parameters()
  def sample(self, shape, rays, bsdf, **kwargs):
    from pytorch3d.pathtracer.utils import (
      cartesian_to_log_polar, param_rusin, weak_sigmoid
    )
    device = rays.device

    sampler = kwargs.get("sampler", self.sampler)
    lights = kwargs.get("lights", self.lights)
    sample_emitter = kwargs.get("sample_emitter_fn", sample_emitter_dir_wo_isect)

    result = torch.zeros(*rays.shape[:-1], 3, device=device)

    it, active = shape.intersect(rays)
    if not active.any(): return result, active, it

    ds, emitter_val = sample_emitter(
      it, shape, lights=lights, sampler=sampler, active=active,
    )
    wo = it.to_local(ds.d)
    result[active] = (1+self.mlp(
      param_rusin(it.n[active], it.wi[active], wo[active]),
      it.p[active],
    ).tanh())/2

    return result, active, it

# Integrator specfic to NeRF for the explicit purpose of optimizing a shape
class NeRFIntegrator(Integrator):
  def dims(self): return self.sub_integrator.dims() + 1
  def __init__(self, sub_integrator, **kwargs):
    super().__init__(**kwargs)
    self.sub_integrator = sub_integrator

  def sample(self, density_field, rays, bsdf, **kwargs):
    result, active, mi = self.sub_integrator.sample(density_field, rays, bsdf, **kwargs)
    alpha = mi.throughput.unsqueeze(-1)
    if mi.with_logits: alpha = alpha.sigmoid()
    result = torch.cat([
      result, alpha,
    ], dim=-1)
    # Now label every point as active
    return result, torch.tensor(True, device=result.device), mi

# A fake integrator which uses nerf instead.
class NeRFReproduce(Integrator):
  def dims(self): return 3
  def __init__(self, **kwargs): super().__init__(**kwargs)
  def sample(self, nerf, rays, lights, **kwargs):
    result = nerf(rays, lights)
    class Dummy:
      ...
    return result, torch.tensor(True, device=result.device), Dummy()

def mis_weight(a, b):
  a *= a
  b = b.square().clamp(min=1e-7)
  return torch.where(a > 0, a/(a+b), torch.zeros_like(b))

# Light Path integrator
class Path(Integrator):
  def __init__(self, training=False, **kwargs):
    super().__init__(**kwargs)
    self.training = training

  def dims(self): return 3
  def sample(self, shapes, rays, bsdf, **kwargs):
    device = rays.device

    sampler = kwargs.get("sampler", self.sampler)
    lights = kwargs.get("lights", self.lights)
    sample_emitter = kwargs.get("sample_emitter_fn", sample_emitter_dir_wo_isect)
    w_isect = kwargs.get("w_isect", False)
    if w_isect is True: sample_emitter = sample_emitter_dir_w_isect
    if type(w_isect) is SkipConnMLP:
      sample_emitter=lambda it,s,lights,sampler,active: \
        sample_emitter_dir_w_learned_occ(it,s,lights,sampler,w_isect,active)

    eta = torch.ones(rays.shape[:-1], device=device)
    # emission_weight = 1.

    throughput = torch.ones(*rays.shape[:-1], 3, device=device)
    result = torch.zeros_like(throughput)

    it, active = shapes.intersect(rays, primary=self.training)

    # break early if there are no active items
    if not active.any(): return result, active, it

    original_active = active.clone()

    #emitter = None
    curr_it = it

    for depth in range(self.max_depth):
      #if emitter is not None:
      #  result[active] += emission_weight * throughput * emitter.eval(interaction, active)

      if depth > self.rr_depth:
        assert(False)
        q = (throughput.max(dim=-1)[0] * eta * eta).clamp(max=0.95)
        active = active & (sampler.sample(q.shape) < q)
        throughput = throughput / q.unsqueeze(-1)

      any_active = active.any()
      if any_active:
        ds, emitter_val = sample_emitter(
          curr_it, shapes,
          lights=lights, sampler=sampler, active=active,
        )
        active_emitted = active & (ds.pdf > 0)
        wo = curr_it.to_local(ds.d) # convert wo to local direction
        bsdf_val, bsdf_pdf = bsdf.eval_and_pdf(curr_it, wo, active=active_emitted)

        mis = torch.ones_like(bsdf_pdf)
        #mis[~ds.delta] = mis_weight(ds.pdf, bsdf_pdf)[~ds.delta]
        result = result + torch.where(
          active_emitted.unsqueeze(-1),
          mis.unsqueeze(-1) * throughput * bsdf_val * emitter_val,
          torch.zeros_like(result),
        )

      bs, bsdf_val = bsdf.sample(curr_it, sampler=sampler, active=active)
      throughput = bsdf_val.clamp(min=1e-10) * throughput
      # detach to save memory
      throughput = throughput.detach()
      active = active & (throughput > 0).any(-1)
      if not active.any(): break

      # eta *= bs.eta # XXX don't currently use eta so no effect

      # spawn rays
      rays = curr_it.spawn_rays(curr_it.from_local(bs.wo))
      curr_it, hits = shapes.intersect(rays, active=active, primary=False)
      active = active & hits
      if not active.any(): break

      # XXX do emitter sampling here? It's fine to omit for now since using only point lights
    #
    return result, original_active, it
