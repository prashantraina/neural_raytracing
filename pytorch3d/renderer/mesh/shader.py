# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import warnings

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from ..blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from ..lighting import PointLights
from ..materials import Materials
from .shading import (
  flat_shading, gouraud_shading, phong_shading,
  neural_shading,
  debug_shading,
)

# A Shader should take as input fragments from the output of rasterization
# along with scene params and output images. A shader could perform operations
# such as:
#     - interpolate vertex attributes for all the fragments
#     - sample colors from a texture map
#     - apply per pixel lighting
#     - blend colors across top K faces per pixel.

FREQUENCY = 2 ** 4

def fourier(tensor, basis) -> torch.Tensor:
  encoding = [tensor]
  # basis is shape (n, 3)
  for i in range(basis.shape[0]):
      inner = torch.sum(basis[i] * tensor, dim=-1, keepdim=True)
      encoding.append(inner.sin())
      encoding.append(inner.cos())

  # Special case, for no positional encoding
  return torch.cat(encoding, dim=-1)

def create_basis(shape, frequency, n_enc_fns, device):
  return torch.normal(0, frequency,
                      size=(n_enc_fns, *shape),
                      dtype=torch.float32,
                      device=device).requires_grad_(False)

# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def rotate_vector(v, axis, c, s):
  return v * c \
         + axis*torch.sum(v * axis, dim=-1, keepdims=True) * (1-c) \
         + torch.cross(axis, v, dim=-1) * s

# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

# http://paulbourke.net/geometry/rotate/
def quat_rot(v, axis, theta):
  q_1 = torch.cat([
    torch.zeros(axis.shape[:-1] + (1,), device=v.device), v,
  ], dim=-1)
  t_2 = (theta/2).expand(axis.shape[:-1] + (1,))
  q_2 = torch.cat([
    t_2.cos(),
    t_2.sin() * axis,
  ], dim=-1)
  q_2_inv = q_2
  q_2_inv[..., 1:] = -q_2_inv[..., 1:]
  out = qmul(qmul(q_2, q_1), q_2_inv)
  return out[..., 1:]

def dir2rusin(n, wi, wo):
    mask = (n != 0).any(-1)
    n =  F.normalize(n.double(), dim=-1) # [ x 3]
    wo = F.normalize(wo.double(), dim=-1) # [ x 3]
    wi = F.normalize(wi.double(), dim=-1) # [ x 3]

    h = F.normalize((wi + wo)/2, dim=-1)
    # comute T'B'H coordinate. h is [0,0,1] in this coordinate
    bp = F.normalize(torch.cross(n, h, dim=-1), dim=-1)
    tp = F.normalize(torch.cross(bp, h, dim=-1), dim=-1)

    # compute theta_d, theta_h, phi_d
    theta_i = torch.sum(wi * n, dim=-1).acos()
    theta_d = torch.sum(h * wi, dim=-1).acos()
    theta_h = torch.sum(h * n, dim=-1).acos()

    # compute visibility
    visiblity = (0 < theta_i) & (theta_i < (math.pi/2))
    mask &= visiblity
    # project the i to the T'B' coordinate
    i_prj = F.normalize(wi - torch.sum(wi*h,dim=-1, keepdims=True)*h, dim=-1)
    # compute the phi_d on the T'B' plane
    cos_phi_d = torch.sum(tp * i_prj, dim=-1).clamp(-1, 1)
    sin_phi_d = torch.sum(torch.cross(tp,i_prj,dim=-1) * n, dim=-1).clamp(-1, 1)
    phi_d = torch.atan2(sin_phi_d, cos_phi_d)
    out = torch.stack([phi_d, theta_h, theta_d], dim=-1)
    # out[~mask, :] = 0
    assert(out.isfinite().all())
    return out.float()

# bases of coordinate system, for finding normal to any vector
e_0 = torch.tensor([1,0,0], device="cuda", dtype=torch.double)
e_1 = torch.tensor([0,1,0], device="cuda", dtype=torch.double)
e_2 = torch.tensor([0,0,1], device="cuda", dtype=torch.double)
def parametrize(n, wo, wi):
  n =  F.normalize(n.double(), dim=-1)
  wo = F.normalize(wo.double(), dim=-1)
  wi = F.normalize(wi.double(), dim=-1)
  # only run this on valid items (those which don't have all 0s in the last dimension)
  mask = (n != 0).any(-1)
  midway = F.normalize((n + e_2)/2, dim=-1)

  # rotate n to [0,0,1] thru midway by by pi here (just assign it since we know the value)
  n[mask] = e_2
  pi = torch.tensor(math.pi, device=n.device)
  wo = rotate_vector(wo, midway, -1, 0)
  # wo = quat_rot(wo, midway, pi)
  wo[~mask] = 0
  wi = rotate_vector(wi, midway, -1, 0)
  # wi = quat_rot(wi, midway, pi)
  wi[~mask] = 0

  # halfway vector between the two light directions
  H = F.normalize((wo + wi)/2, dim=-1)

  theta_h = H[..., 2].acos()
  phi_h = torch.atan2(H[..., 1], H[..., 0])

  binormal = e_1

  v = -phi_h[..., None]
  # is this wo or wi?
  tmp = F.normalize(rotate_vector(wi, n, v.cos(), v.sin()), dim=-1)
  # tmp = F.normalize(quat_rot(wi, n, v), dim=-1)

  v = -theta_h[..., None]
  diff = F.normalize(
    rotate_vector(tmp, binormal.expand(tmp.shape), v.cos(), v.sin()),
    # quat_rot(tmp, binormal.expand(tmp.shape), v),
    dim=-1,
  )

  # clamp to remove NaN
  theta_d = diff[..., 2].clamp(0, 1).acos()
  phi_d = torch.fmod(torch.atan2(diff[..., 1], diff[..., 0]), math.pi)
  assert(((phi_d >= -math.pi) & (phi_d <= math.pi)).all())

  out = torch.stack([phi_d, theta_h, theta_d], dim=-1)
  # Zero out invalid elements
  out[~mask, :] = 0
  assert(out.isfinite().all())
  return out.float()


class MiniMLP(torch.nn.Module):
  def __init__(
    self,

    n,
    out=3,

    num_layers=4,
    hidden_size=64,
    skip_connect_every=3,

    device="cuda",
  ):
    super(MiniMLP, self).__init__()
    self.n = n
    self.layer1 = torch.nn.Linear(n, hidden_size)
    self.skip_connect_every = skip_connect_every
    self.layers = torch.nn.ModuleList()
    for i in range(num_layers):
      if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
        self.layers.append(torch.nn.Linear(n + hidden_size, hidden_size))
      else:
        self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
    self.last = torch.nn.Linear(hidden_size, out)

    self.relu = torch.nn.functional.relu
  def forward(self, x):
    original = x

    # since we're using all the items we can just pass it thru
    x = self.layer1(x)
    for i, layer in enumerate(self.layers):
      if (i % self.skip_connect_every == 0 and i > 0 and i != len(self.layers) - 1):
        x = torch.cat((original, x), dim=-1)
      x = self.relu(layer(x))
    return self.last(x)

class NeuralParametrizedBSDF3(torch.nn.Module):
  def __init__(
    self,

    # number of linear layers
    # WORKS with 8 layers, 64 hidden-size
    num_layers=8,

    hidden_size=64,

    skip_connect_every=3,

    # num per encoding
    n_per_enc=1,

    # num encodings for uv (disabled when both are false)
    n_uv=4,
    w_uv=True,

    # num encodings for wi
    n_wi=4,
    w_wi=True,

    # num encodings for wo
    n_wo=4,
    w_wo=True,

    # num encodings for normal
    n_n=4,
    w_n=True,

    n_p=4,
    w_p=True,

    n_angles=4,
    w_angles=True,

    device="cuda",
  ):
    super(NeuralParametrizedBSDF3, self).__init__()

    w_uv = 2 if w_uv else 0
    self.w_uv = w_uv

    w_p  = 3 if w_p else 0
    self.w_p = w_p

    w_wi = 3 if w_wi else 0
    self.w_wi = w_wi

    w_wo = 3 if w_wo else 0
    self.w_wo = w_wo

    w_n  = 3 if w_n else 0
    self.w_n = w_n

    w_angles = 3 if w_angles else 0
    self.w_angles = w_angles

    self.basis_uv = create_basis([w_uv], FREQUENCY, n_uv, device)
    self.basis_p = create_basis([w_p], FREQUENCY, n_p, device)
    self.basis_n = create_basis([w_n], FREQUENCY, n_n, device)
    self.basis_wi = create_basis([w_wi], FREQUENCY, n_wi, device)
    self.basis_wo = create_basis([w_wo], FREQUENCY, n_wo, device)
    self.basis_angles = create_basis([w_angles], FREQUENCY, n_angles, device)

    self.dim_uv = w_uv + 2 * n_per_enc * n_uv
    self.dim_p = w_p + 2 * n_per_enc * n_p
    self.dim_n  = w_wo + 2 * n_per_enc * n_n
    self.dim_wi = w_wi + 2 * n_per_enc * n_wi
    self.dim_wo = w_wo + 2 * n_per_enc * n_wo
    self.dim_angles = w_angles + 2 * n_per_enc * n_angles
    #total = self.dim_uv + self.dim_p + self.dim_n + \
    #  self.dim_wo + self.dim_wi + self.dim_angles
    total = self.dim_p + self.dim_angles

    self.layer1 = torch.nn.Linear(total, hidden_size)

    self.layers_xyz = torch.nn.ModuleList()
    self.skip_connect_every = skip_connect_every
    for i in range(num_layers):
      if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
        self.layers_xyz.append(torch.nn.Linear(total + hidden_size, hidden_size))
      else:
        self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

    self.relu = torch.nn.functional.relu

    self.spectrum_layers = torch.nn.Sequential(
      torch.nn.Linear(total + hidden_size, hidden_size // 2),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_size // 2, hidden_size // 2),
      torch.nn.ReLU(),
    )

    self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
    self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
    self.fc_alpha = torch.nn.Linear(hidden_size, 1)


  def forward(self, x):
    assert(x.isfinite().all())
    uv, p, n, wo, wi = torch.split(x,
      [self.w_uv, self.w_p, self.w_n, self.w_wo, self.w_wi], dim=-1)
    x = torch.cat([
      # fourier(uv, self.basis_uv),
      fourier(p, self.basis_p),
      # fourier(n, self.basis_n),
      # fourier(wi, self.basis_wi),
      # fourier(wo, self.basis_wo),
      fourier(parametrize(n, wi, wo), self.basis_angles),
    ], dim=-1)

    original = x

    # since we're using all the items we can just pass it thru
    x = self.layer1(x)
    for i, layer in enumerate(self.layers_xyz):
      if (
        i % self.skip_connect_every == 0
        and i > 0
        and i != len(self.layers_xyz) - 1
      ):
        x = torch.cat((x, original), dim=-1)
      x = self.relu(layer(x))

    feat = self.relu(self.fc_feat(x))
    alpha = torch.sigmoid(self.fc_alpha(x))

    x = torch.cat((feat, original), dim=-1)

    x = self.spectrum_layers(x)
    # for l in self.spectrum_layers: x = self.relu(l(x))

    rgb = self.fc_rgb(x)
    # XXX Sigmoid is optional, but have to see if it makes it better or worse
    # rgb = torch.sigmoid(rgb)

    return rgb * alpha

class NeuralParametrizedBSDF(torch.nn.Module):
  def __init__(
    self,

    num_layers=8,
    hidden_size=64,
    skip_connect_every=3,

    n_per_enc=1,
    n_uv=4,
    w_uv=True,

    n_wi=4,
    w_wi=True,

    n_wo=4,
    w_wo=True,

    n_n=4,
    w_n=True,

    n_angles=4,
    w_angles=True,

    device="cuda",
  ):
    super(NeuralParametrizedBSDF, self).__init__()

    w_uv = 2 if w_uv else 0
    self.w_uv = w_uv

    w_angles = 3 if w_angles else 0
    self.w_angles = w_angles

    self.basis_uv = create_basis([w_uv], 2**4, n_uv, device)
    self.basis_angles = create_basis([w_angles], 2**4, n_angles, device)

    self.dim_uv = w_uv + 2 * n_per_enc * n_uv
    self.dim_angles  = w_angles + 2 * n_per_enc * n_angles
    total = self.dim_uv + self.dim_angles

    self.layer1 = torch.nn.Linear(total, hidden_size)

    self.layers_xyz = torch.nn.ModuleList()
    self.skip_connect_every = skip_connect_every
    for i in range(num_layers):
      if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
        self.layers_xyz.append(torch.nn.Linear(total + hidden_size, hidden_size))
      else:
        self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

    self.relu = torch.nn.functional.relu
    self.spectrum_layers = torch.nn.Sequential(
      torch.nn.Linear(total + hidden_size, hidden_size // 2),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_size // 2, hidden_size // 2),
      torch.nn.ReLU(),
    )

    self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
    self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
    self.fc_alpha = torch.nn.Linear(hidden_size, 1)


  def forward(self, x):
    assert(x.isfinite().all())
    uv, n, wo, wi = torch.split(x, [self.w_uv, 3, 3, 3], dim=-1)
    angles = dir2rusin(n, wo, wi)

    x = torch.cat([
      fourier(uv, self.basis_uv),
      fourier(n, self.basis_n),
      fourier(wo, self.basis_wo),
      fourier(wi, self.basis_wi),
      fourier(angles, self.basis_angles)
    ], dim=-1)

    original = x

    # since we're using all the items we can just pass it thru
    x = self.layer1(x)
    for i, layer in enumerate(self.layers_xyz):
      if (i % self.skip_connect_every == 0 and i > 0 and i != len(self.layers_xyz) - 1):
        x = torch.cat((x, original), dim=-1)
      x = self.relu(layer(x))

    feat = self.relu(self.fc_feat(x))
    alpha = self.fc_alpha(x)
    x = torch.cat((feat, original), dim=-1)

    x = self.spectrum_layers(x)
    rgb = self.fc_rgb(x)
    return rgb * alpha

class NeuralParametrizedBSDF2(torch.nn.Module):
  def __init__(
    self,

    n_per_enc=1,
    n_uv=8,
    w_uv=True,

    n_fi=4,
    w_fi=True,

    n_angles=4,
    w_angles=True,

    device="cuda",
  ):
    super(NeuralParametrizedBSDF2, self).__init__()

    w_uv = 2 if w_uv else 0
    self.w_uv = w_uv

    w_fi = 1 if w_fi else 0
    self.w_fi = w_fi

    w_angles = 3 if w_angles else 0
    self.w_angles = w_angles

    self.basis_uv = create_basis([w_uv], 2**4, n_uv, device)
    self.basis_fi = create_basis([w_fi], 2**4, n_fi, device)
    self.basis_angles = create_basis([w_angles], 2**4, n_angles, device)

    self.dim_uv = w_uv + 2 * n_per_enc * n_uv
    self.dim_fi = w_fi + 2 * n_per_enc * n_fi

    self.dim_angles  = w_angles + 2 * n_per_enc * n_angles
    print(self.dim_uv + self.dim_fi)

    # should be purely concerned with color component
    self.mlp_uv = MiniMLP(
      n=self.dim_uv + self.dim_fi,
      num_layers=24,
      hidden_size=16,
    )
    # should be purely concerned with lighting component
    self.mlp_angles = MiniMLP(
      self.dim_angles,
      num_layers=4,
      hidden_size=64,
    )

  def forward(self, x):
    assert(x.isfinite().all())
    uv, fi, n, wo, wi = torch.split(x, [self.w_uv, self.w_fi, 3, 3, 3], dim=-1)
    angles = parametrize(n, wo, wi)

    c = torch.cat([
      fourier(uv, self.basis_uv),
      fourier(fi, self.basis_fi)
    ], dim=-1)

    # different components of color
    return self.mlp_uv(c) * tself.mlp_angles(fourier(angles, self.basis_angles))

class NeuralBSDF(torch.nn.Module):
  def __init__(
    self,

    # number of linear layers
    # WORKS with 8 layers, 64 hidden-size
    num_layers=8,

    hidden_size=64,

    skip_connect_every=3,

    # num per encoding
    n_per_enc=1,

    # num encodings for uv (disabled when both are false)
    n_uv=4,
    w_uv=True,

    # num encodings for wi
    n_wi=4,
    w_wi=True,

    # num encodings for wo
    n_wo=4,
    w_wo=True,

    # num encodings for normal
    n_n=4,
    w_n=True,

    device="cuda",
  ):
    super(NeuralBSDF, self).__init__()

    w_uv = 2 if w_uv else 0
    self.w_uv = w_uv


    w_wi = 3 if w_wi else 0
    self.w_wi = w_wi

    w_wo = 3 if w_wo else 0
    self.w_wo = w_wo

    w_n  = 3 if w_n  else 0
    self.w_n = w_n

    self.basis_uv = create_basis([w_uv], FREQUENCY, n_uv, device)
    self.basis_n = create_basis([w_n], FREQUENCY, n_n, device)
    self.basis_wi = create_basis([w_wi], FREQUENCY, n_wi, device)
    self.basis_wo = create_basis([w_wo], FREQUENCY, n_wo, device)

    self.dim_uv = w_uv + 2 * n_per_enc * n_uv
    self.dim_n  = w_wo + 2 * n_per_enc * n_n
    self.dim_wi = w_wi + 2 * n_per_enc * n_wi
    self.dim_wo = w_wo + 2 * n_per_enc * n_wo
    total = self.dim_uv + self.dim_n + self.dim_wo + self.dim_wi

    self.layer1 = torch.nn.Linear(total, hidden_size)

    self.layers_xyz = torch.nn.ModuleList()
    self.skip_connect_every = skip_connect_every
    for i in range(num_layers):
      if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
        self.layers_xyz.append(torch.nn.Linear(total + hidden_size, hidden_size))
      else:
        self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

    self.relu = torch.nn.functional.relu

    self.spectrum_layers = torch.nn.Sequential(
      torch.nn.Linear(total + hidden_size, hidden_size // 2),
      self.relu,
      torch.nn.Linear(hidden_size // 2, hidden_size // 2),
      self.relu,
    )

    self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
    self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
    self.fc_alpha = torch.nn.Linear(hidden_size, 1)


  def forward(self, x):
    assert(x.isfinite().all())
    uv, _, n, wo, wi = torch.split(
      x,
      [self.w_uv, 1, self.w_n, self.w_wo, self.w_wi],
      dim=-1,
    )
    x = torch.cat([
      fourier(uv, self.basis_uv),
      fourier(n, self.basis_n),
      fourier(wi, self.basis_wi),
      fourier(wo, self.basis_wo),
    ], dim=-1)

    original = x

    # since we're using all the items we can just pass it thru
    x = self.layer1(x)
    for i, layer in enumerate(self.layers_xyz):
      if (
        i % self.skip_connect_every == 0
        and i > 0
        and i != len(self.layers_xyz) - 1
      ):
        x = torch.cat((x, original), dim=-1)
      x = self.relu(layer(x))

    feat = self.relu(self.fc_feat(x))
    alpha = torch.sigmoid(self.fc_alpha(x))

    x = torch.cat((feat, original), dim=-1)

    x = self.spectrum_layers(x)
    # for l in self.spectrum_layers: x = self.relu(l(x))

    rgb = self.fc_rgb(x)
    # XXX Sigmoid is optional, but have to see if it makes it better or worse
    # rgb = torch.sigmoid(rgb)

    return rgb * alpha

class NeuralShader(nn.Module):
    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.NN = NeuralParametrizedBSDF3(device=device).to(device)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)

        texels, pixels_uv = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = neural_shading(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            pixels_uv=pixels_uv,

            NN=self.NN,
            texels = texels,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images

class HardPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)

        texels, _ = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images


class SoftPhongShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftPhongShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


class HardGouraudShader(nn.Module):
    """
    Per vertex lighting - the lighting model is applied to the vertex colors and
    the colors are then interpolated using the barycentric coordinates to
    obtain the colors for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardGouraudShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardGouraudShader"
            raise ValueError(msg)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)

        # As Gouraud shading applies the illumination to the vertex
        # colors, the interpolated pixel texture is calculated in the
        # shading step. In comparison, for Phong shading, the pixel
        # textures are computed first after which the illumination is
        # applied.
        pixel_colors = gouraud_shading(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(pixel_colors, fragments, blend_params)
        return images


class SoftGouraudShader(nn.Module):
    """
    Per vertex lighting - the lighting model is applied to the vertex colors and
    the colors are then interpolated using the barycentric coordinates to
    obtain the colors for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = SoftGouraudShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftGouraudShader"
            raise ValueError(msg)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        pixel_colors = gouraud_shading(
            meshes=meshes,
            fragments=fragments,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            pixel_colors, fragments, self.blend_params, znear=znear, zfar=zfar
        )
        return images


def TexturedSoftPhongShader(
    device="cpu", cameras=None, lights=None, materials=None, blend_params=None
):
    """
    TexturedSoftPhongShader class has been DEPRECATED. Use SoftPhongShader instead.
    Preserving TexturedSoftPhongShader as a function for backwards compatibility.
    """
    warnings.warn(
        """TexturedSoftPhongShader is now deprecated;
            use SoftPhongShader instead.""",
        PendingDeprecationWarning,
    )
    return SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        materials=materials,
        blend_params=blend_params,
    )


class HardFlatShader(nn.Module):
    """
    Per face lighting - the lighting model is applied using the average face
    position and the face normal. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardFlatShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = flat_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images


class SoftSilhouetteShader(nn.Module):
    """
    Calculate the silhouette by blending the top K faces for each pixel based
    on the 2d euclidean distance of the center of the pixel to the mesh face.

    Use this shader for generating silhouettes similar to SoftRasterizer [0].

    .. note::

        To be consistent with SoftRasterizer, initialize the
        RasterizationSettings for the rasterizer with
        `blur_radius = np.log(1. / 1e-4 - 1.) * blend_params.sigma`

    [0] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
        3D Reasoning', ICCV 2019
    """

    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """
        Only want to render the silhouette so RGB values can be ones.
        There is no need for lighting or texturing
        """
        colors = torch.ones_like(fragments.bary_coords)
        blend_params = kwargs.get("blend_params", self.blend_params)
        images = sigmoid_alpha_blend(colors, fragments, blend_params)
        return images

class DebugShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        return debug_shading(meshes, fragments)
