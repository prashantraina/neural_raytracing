import torch
from .integrators import Path
from .samplers import Sampler
from .bsdf import Diffuse
from .scene import ( mesh_intersect, mesh_intersect_test )
from .utils import rand_uv
from tqdm import trange
from torchvision.transforms.functional import rotate

def nothing(_): return None

# Entry point to path tracing
def pathtrace(
  shapes,
  lights,
  cameras,
  integrator,
  bsdf=None,

  # how big should the image be
  size=512,
  width=None,
  height=None,
  # how large should chunks be processed at a time
  chunk_size=32,
  # how many rays should be processed in parallel
  bundle_size=4,
  # what color is used in the background
  background = 1,
  addition=nothing,
  sampler=Sampler(),
  silent= False,
  # this specifies some extra space to be rendered, so that edge effects are reduced.
  trim=0,
  device="cuda",
  squeeze_first=True,
  w_isect=False,
  with_noise=1e-3,
):
  batch_dims = len(cameras)
  if width is None: width = size
  if height is None: height = size
  # RGB destination for output images (all gray to easily see white or black)
  output_images = torch.full(
    [batch_dims, width, height, integrator.dims()],
    background,
    device=device,
    dtype=torch.float,
  )

  view = lambda v: v
  if trim != 0: view = lambda v: v[trim:-trim, trim:-trim]

  assert((size % chunk_size) == 0),\
    f"Can only specify chunk sizes which evenly divide size, {size} % {chunk_size}"

  x = torch.arange(start=0, end=width, step=chunk_size, device=device)
  y = torch.arange(start=0, end=height, step=chunk_size, device=device)

  # TODO https://pytorch.org/docs/stable/notes/multiprocessing.html
  grid_x, grid_y = torch.meshgrid(x, y)
  iterator = (range if silent else trange)(len(x)*len(y))
  for ij in iterator:
    i, j = divmod(ij, len(y))
    x_start, y_start = grid_x[j, i], grid_y[j, i]
    sub_g_x, sub_g_y = torch.meshgrid(
      torch.arange(x_start-trim, x_start+chunk_size+trim, device=device, dtype=torch.float),
      torch.arange(y_start-trim, y_start+chunk_size+trim, device=device, dtype=torch.float),
    )

    positions = torch.stack([sub_g_y, sub_g_x], dim=-1)

    # generate rays in shape (W, H, bundle_size, 6)
    rays = cameras.sample_positions(
      positions, sampler, bundle_size, size=size,
      N=batch_dims, with_noise=with_noise,
    )

    values, mask, it = integrator.sample(
      shapes, rays,
      bsdf=bsdf, lights=lights, sampler=sampler,
      w_isect=w_isect,
    )

    valid_pixels = mask.any(dim=-1) # valid along bundles
    # compute means along the sampling dimension
    v = torch.mean(values, dim=-2)
    # set invalid items as background XXX caution if ever write to pixels multiple times
    v[~valid_pixels] = background
    output_images[:, x_start:x_start+chunk_size, y_start:y_start+chunk_size, :] = view(v)

  if squeeze_first and (batch_dims == 1): output_images = output_images.squeeze(0)
  return output_images, addition(it)

# Sample some region of the image-space with pathtracing. Useful when the entire image is too
# big to be rendered in one pass or only training on a small sample of the image.
def pathtrace_sample(
  shapes,
  lights,
  cameras,
  integrator,
  bsdf=None,

  # how big should the image be
  size=512,
  # how large should chunks be processed at a time of each crop
  chunk_size=32,
  # how many rays should be processed in parallel
  bundle_size=4,
  # what is the total size of the visible region
  crop_size=128,
  # which coordinate is the top left? (int, int)
  uv=None,
  # what color is used in the background
  background = 1,
  sampler=Sampler(),
  addition=nothing,
  silent= False,
  mode="crop",
  device="cuda",
  squeeze_first=True,
  w_isect=False,
  with_noise=1e-2,
):
  if uv is None: uv = rand_uv(size, size, crop_size)
  batch_dims = len(cameras)
  # RGB destination for output images
  img_size = [batch_dims, size, size, integrator.dims()]
  if mode == "crop": img_size = [batch_dims, crop_size, crop_size, integrator.dims()]
  output_images = torch.full(img_size, background, device=device, dtype=torch.float)


  assert((size % chunk_size) == 0),\
    f"Can only specify chunk sizes which evenly divide size, {size} % {chunk_size}"
  chunk_size = min(chunk_size, crop_size)

  u = max(min(uv[0], size-crop_size), 0)
  v = max(min(uv[1], size-crop_size), 0)
  x = torch.arange(start=u, end=u+crop_size, step=chunk_size, device=device)
  y = torch.arange(start=v, end=v+crop_size, step=chunk_size, device=device)

  def update_full(vals, xs, ys):
    output_images[:, xs:xs+chunk_size, ys:ys+chunk_size] = vals
  def update_crop(vals, xs, ys):
    output_images[:, xs-u:xs-u+chunk_size, ys-v:ys-v+chunk_size] = vals

  update = update_full
  if mode == "crop": update = update_crop

  grid_x, grid_y = torch.meshgrid(x, y)
  iterator = (range if silent else trange)(len(x)*len(y))
  for ij in iterator:
    i, j = divmod(ij, len(y))
    x_start, y_start = grid_x[j, i], grid_y[j, i]
    sub_g_x, sub_g_y = torch.meshgrid(
      torch.arange(x_start, x_start+chunk_size, device=device, dtype=torch.float),
      torch.arange(y_start, y_start+chunk_size, device=device, dtype=torch.float),
    )

    positions = torch.stack([sub_g_y, sub_g_x], dim=-1)

    rays = cameras.sample_positions(
      positions, sampler, bundle_size, size=size,
      N=batch_dims, with_noise=with_noise,
    )

    values, mask, it = integrator.sample(
      shapes, rays, bsdf=bsdf, lights=lights, sampler=sampler,
      w_isect=w_isect,
    )

    valid_pixels = mask.any(dim=-1) # valid along bundles
    vals = torch.mean(values, dim=-2)
    vals[~valid_pixels] = background
    update(vals, x_start, y_start)

  if squeeze_first and (batch_dims == 1): output_images = output_images.squeeze(0)
  setattr(it, 'crop_uv', uv)
  return output_images, addition(it)
