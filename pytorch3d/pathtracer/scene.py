import torch
import torch.nn.functional as F
import math
from .interaction import SurfaceInteraction
from .utils import dir_to_elev_azim

# EPSILON for intersection
EPS = 1e-9

def mesh_intersect(
  meshes,
  rays, # Ray(W, H, Bundle, 6)

  max_t = math.inf,
  split_n = 256,
  active=True,
) -> torch.Tensor:
  device = rays.device
  r_o, r_d = torch.split(rays, 3, dim=-1)

  out_active = torch.zeros(r_o.shape[:-1], dtype=torch.bool, device=device) & active

  # best_faces = torch.zeros_like(out_active, dtype=torch.long, device=device)
  best_dists = torch.full_like(out_active, math.inf, dtype=torch.float, device=device)
  uvs = torch.zeros(out_active.shape + (2,), dtype=torch.float, device=device)
  normals = torch.zeros(out_active.shape + (3,), dtype=torch.float, device=device)

  r_o_r = r_o.expand(split_n, *r_o.shape)
  r_d_r = r_d.expand_as(r_o_r)

  verts = meshes.verts_packed() # (V, 3)
  faces = meshes.faces_packed() # (F, 3)
  fv = verts[faces] # (F, 3 verts per face, 3 coords per vert)

  for fv in torch.split(fv, split_n, dim=0):
    batch_size = fv.shape[0]
    r_o_r, r_d_r = r_o_r[:fv.shape[0]], r_d_r[:fv.shape[0]] # in case of last iter being small
    e_1 = (fv[:, 1] - fv[:, 0])[:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)\
      .transpose(0, 1)
    e_2 = (fv[:, 2] - fv[:, 0])[:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)\
      .transpose(0, 1)
    v_0 = fv[:, 0][:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)\
      .transpose(0, 1)

    active = torch.ones(e_2.shape[:-1], dtype=torch.bool, device=device)

    h = torch.cross(r_d_r, e_2, dim=-1)
    a = torch.sum(e_1 * h, dim=-1)
    active &= (a < -EPS) | (a > EPS)

    f = (a+1e-7).reciprocal()
    s = r_o_r - v_0
    u = f * torch.sum(s * h, dim=-1)
    active &= (u >= 0) & (u <= 1)

    q = torch.cross(s, e_1)
    v = f * torch.sum(r_d_r*q, dim=-1)
    active &= (v >= 0) & (u + v <= 1)

    t = f * torch.sum(e_2 * q, dim=-1)
    active &= (t > EPS) & (t < (max_t - EPS))

    # For each ray, is there any valid face(dim 0)
    valid_mins = active.any(0)
    # if no rays are valid just skip this next block
    if not valid_mins.any(): continue
    # if any faces are valid, then there must've been an intersection so need to mark valid
    out_active |= valid_mins

    # which faces are valid?
    t[~active] = math.inf

    # find which t-value should be selected for each ray
    min_t, faces = t.min(dim=0)
    lesser = best_dists > min_t

    # only replace those which are valid and that have their min_t smaller than the current_best
    replace_cond = valid_mins & lesser

    best_dists[replace_cond] = min_t[replace_cond]

    frc = faces[replace_cond]

    u = u[frc, replace_cond]
    #assert(((u >= 0) & (u <= 1)).all())
    v = v[frc, replace_cond]
    #assert(((v >= 0) & (u + v <= 1)).all())
    w = 1 - u - v
    # assert(((w >= 0) & (w <= 1)).all()) # This should always be true.

    uvs[replace_cond] = torch.stack([u, v], dim=-1)

    # THIS IS THE CORRECT PERMUTATION
    bary = torch.stack([w, v, u], dim=-1)
    # normals[replace_cond] = bary # display barycentric
    # TODO add conversion which also uses the pixel uv coords maybe (x * 2)-1?
    #normals[replace_cond] = (bary[..., None] * fvn[frc]).sum(dim=-2)
    normals[replace_cond] = F.normalize(
      torch.cross(e_1[frc, replace_cond], e_2[frc, replace_cond], dim=-1),
      dim=-1,
    )


  out = SurfaceInteraction.zeros(out_active.shape + (3,), device=device)
  out.p[out_active] = r_o[out_active] + best_dists[..., None][out_active] * r_d[out_active]
  # just move a little bit off of the surface for stability in future rays.
  out.p[out_active] += normals[out_active] * 1e-5

  out.t = best_dists
  out.uv = uvs
  out.set_normals(normals)
  out.wi = out.to_local(-r_d)
  # returns interaction<H, W, Bundle_size>, active
  return out, out_active

def mesh_intersect_test(
  meshes,
  rays, # Ray(W, H, Bundle, 6)

  max_t = math.inf,
  split_n = 128,
) -> torch.Tensor:
  device = rays.device

  r_o, r_d = torch.split(rays, 3, dim=-1)

  out_active = torch.zeros(r_o.shape[:-1], dtype=torch.bool, device=device)

  r_o_r = r_o.expand(split_n, *r_o.shape)
  r_d_r = r_d.expand_as(r_o_r)

  verts = meshes.verts_packed() # (V, 3)
  faces = meshes.faces_packed() # (F, 3)
  fv = verts[faces] # (F, 3 verts per face, 3 coords per vert)

  for fv in torch.split(fv, split_n, dim=0):
    split_n = fv.shape[0]
    r_o_r, r_d_r = r_o_r[:fv.shape[0]], r_d_r[:fv.shape[0]] # in case of last one being shorter
    e_1 = (fv[:, 1] - fv[:, 0])[:, None, None, None].expand(split_n, *r_o.shape)
    e_2 = (fv[:, 2] - fv[:, 0])[:, None, None, None].expand(split_n, *r_o.shape)
    v_0 = fv[:, 0][:, None, None, None].expand(split_n, *r_o.shape)

    active = torch.ones(e_2.shape[:-1], dtype=torch.bool, device=device)

    h = torch.cross(r_d_r, e_2, dim=-1)
    a = torch.sum(e_1 * h, dim=-1)
    active &= (a < -EPS) | (a > EPS)

    f = (a+1e-7).reciprocal()
    s = r_o_r - v_0
    u = f * torch.sum(s * h, dim=-1)
    active &= (u >= 0) & (u <= 1)

    q = torch.cross(s, e_1)
    v = f * torch.sum(r_d_r*q, dim=-1)
    active &= (v >= 0) & (u + v <= 1)

    t = f * (e_2 * q).sum(dim=-1)
    active &= (t > EPS) & (t < (max_t - EPS))
    out_active |= active.any(0)

  return out_active


# Finds the shortest distance between rays p & semgnts q
# FIXME I way over thought this, I can omit all the extra indexing, that's handled by earlier
# dimensions.
# http://geomalgorithms.com/a07-_distance.html
def distance_between_rays_and_segments(p_o: [..., 3], p_r: [..., 3], q_o: [..., 3], q_r: [..., 3]):
  # compute s_c and t_c which are lengths along p and q
  w_o = p_o - q_o
  a = (p_r * p_r).sum(dim=-1)
  b = (p_r * q_r).sum(dim=-1)
  c = (q_r * p_r).sum(dim=-1)

  d = (p_r * w_o).sum(dim=-1)
  e = (q_r * w_o).sum(dim=-1)

  s_d = a * c - b * b
  t_d = s_d.clone()
  s_n = (b*e - c*d)
  t_n = (a*e - b*d)

  parll = (s_d == 0) # parallel but persnickety
  s_n[parll] = 0
  s_d[parll] = 1
  t_n[parll] = e[parll]
  t_d[parll] = c[parll]

  # Could omit the following if it was just infinite lines
  short_mask = t_n < 0

  t_n[short_mask] = 0
  low_mask = (-d < 0) & short_mask
  s_n[low_mask] = 0
  mix_mask = (-d > a) & short_mask
  s_n[mix_mask] = s_d[mix_mask]
  else_mask = (~low_mask) & (~mix_mask)
  s_n[else_mask] = -d[else_mask]
  s_d[else_mask] = a[else_mask]

  long_mask = t_n > t_d

  t_n[long_mask] = t_d[long_mask]
  b_m_d = b - d
  low_mask = (b_m_d < 0) & long_mask
  s_n[low_mask] = 0
  mix_mask = (b_m_d > a) & long_mask
  s_n[mix_mask] = s_d[mix_mask]
  else_mask = (~low_mask) & (~mix_mask)
  s_n[else_mask] = b_m_d[else_mask]
  s_d[else_mask] = a[else_mask]

  s_c = s_n/s_d
  t_c = t_n/t_d

  v = (s_c * p_r - t_c * q_r)
  return torch.norm(w_o + v, dim=-1)

def ray_point_dist(r_o_r, r_d_r, v: [..., 3]):
  m2 = (r_d_r * r_d_r).sum(-1, keepdim=True)
  q = v - r_o_r
  t_0 = (r_d_r * q).sum(-1, keepdim=True)/m2

  p_closest = r_o_r + r_d_r * t_0
  dists = (v - p_closest).abs()
  return torch.norm(dists, dim=-1)

# uses a blend of sphere marching and mesh intersection(projected in the direction of the ray)
# to compute minimum distance to a surface
def mesh_level_surfaces(
  meshes,
  rays, # Ray(W, H, Bundle, 6)

  max_t = 10,
  split_n = 128,
):
  device = rays.device
  r_o, r_d = torch.split(rays, 3, dim=-1)

  best_dists = torch.full(r_o.shape[:-1], max_t, dtype=torch.float, device=device)

  r_o_r = r_o.expand(split_n, *r_o.shape)
  r_d_r = r_d.expand_as(r_o_r)

  verts = meshes.verts_packed() # (V, 3)
  faces = meshes.faces_packed() # (F, 3)
  fv = verts[faces] # (F, 3 verts per face, 3 coords per vert)

  for fv in torch.split(fv, split_n, dim=0):
    batch_size = fv.shape[0]
    r_o_r, r_d_r = r_o_r[:fv.shape[0]], r_d_r[:fv.shape[0]] # in case of last one being shorter
    e_1 = (fv[:, 1] - fv[:, 0])[:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)
    e_2 = (fv[:, 2] - fv[:, 0])[:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)
    v_0 = fv[:, 0][:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)

    hits = torch.ones(e_2.shape[:-1], dtype=torch.bool, device=device)

    h = torch.cross(r_d_r, e_2, dim=-1)
    a = torch.sum(e_1 * h, dim=-1)
    hits &= (a < -EPS) | (a > EPS)

    f = (a+1e-7).reciprocal()
    s = r_o_r - v_0
    u = f * torch.sum(s * h, dim=-1)
    hits &= (u >= 0) & (u <= 1)

    q = torch.cross(s, e_1)
    v = f * torch.sum(r_d_r*q, dim=-1)
    hits &= (v >= 0) & (u + v <= 1)

    t = f * torch.sum(e_2 * q, dim=-1)
    hits &= (t > EPS) & (t < (max_t - EPS))

    valid_mins = hits.any(0) # For each ray, is there any valid face(dim 0)
    # any ray that hits something is 0 (no distance from surface)
    best_dists[valid_mins] = 0

    v_0_dists = ray_point_dist(r_o_r, r_d_r, v_0).min(0)[0]
    v_1 = fv[:, 1][:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)
    v_2 = fv[:, 2][:, None, None, None].repeat(1, *r_d_r.shape[1:-1], 1)
    best_dists[~valid_mins] = torch.min(v_0_dists[~valid_mins], best_dists[~valid_mins])
    v_1_dists = ray_point_dist(r_o_r, r_d_r, v_1).min(0)[0]
    best_dists[~valid_mins] = torch.min(v_1_dists[~valid_mins], best_dists[~valid_mins])
    v_2_dists = ray_point_dist(r_o_r, r_d_r, v_2).min(0)[0]
    best_dists[~valid_mins] = torch.min(v_2_dists[~valid_mins], best_dists[~valid_mins])
  return best_dists


def sample_emitter_dir_w_isect(it, shapes, lights, sampler, active=True):
  ds, spectrum = lights.sample_direction(it, sampler=sampler, active=active)

  # ds.d is already in world space
  rays = torch.cat([it.p, ds.d], dim=-1)
  not_blocked = \
    shapes.intersect_test(rays, max_t=ds.dist.reshape_as(active)[..., None], active=active)
  spectrum[~not_blocked | ~active] = 0
  return ds, spectrum

# Add a simple learned occlusion amount to compensate for doing direct lighting
def sample_emitter_dir_w_learned_occ(it, shapes, lights, sampler, occ, active=True):
  ds, spectrum = lights.sample_direction(it, sampler=sampler, active=active)

  # ds.d is already in world space
  rays = torch.cat([it.p, ds.d], dim=-1)
  not_blocked = \
    shapes.intersect_test(rays, max_t=ds.dist.reshape_as(active)[..., None], active=active)
  occluded = ~not_blocked
  occ_rays = torch.cat([
    it.p,
    dir_to_elev_azim(ds.d),
  ], dim=-1)
  spectrum = torch.where(
    occluded[..., None],
    occ(occ_rays).sigmoid() * spectrum,
    spectrum,
  )
  spectrum = active[..., None] * spectrum
  return ds, spectrum

def sample_emitter_dir_wo_isect(it, shapes, lights, sampler, active=True):
  ds, spectrum = lights.sample_direction(it, sampler=sampler, active=active)
  spectrum[~active] = 0
  return ds, spectrum






