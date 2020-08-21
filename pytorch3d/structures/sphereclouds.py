import torch

class SphereClouds(object):
  _INTERNAL_TENSORS = [
    "_points",
    "_normals",
    "_features",
  ]
  def __init__(self, points, normals=None, features=None):
    self.device = None
    self.equisized = False
    self.valid = None
    self._N = 0
    self._P = 0
    self._C = None

    self._points = None
    self._normals = None
    self._features = None
    assert(torch.is_tensor(points))
    assert(points.dim() == 3)
    shape = points.shape
    assert(shape[-1] == 4)

    self._points = points
    self._N, self._P = shape[:2]

    self.device = points.device

    self.valid = torch.ones([self._N], dtype=torch.bool, device=self.device)
    self._num_points_per_cloud = torch.tensor(
      [self._P] * self._N, device=self.device
    )
    self.equisized = True
  def clone(self, fn):
    points, normals, features = None, None, None
    if self._points is not None: points = self._points.clone()
    if self._normals is not None: normals = self._normals.clone()
    if self._features is not None: features = self._normals.clone()
    return self.__class__(points=points, normals=normals, features=features)
  def detach(self):
    points, normals, features = None, None, None
    if self._points is not None: points = self._points.detach()
    if self._normals is not None: normals = self._normals.detach()
    if self._features is not None: features = self._normals.detach()
    other = self.__class__(points=points, normals=normals, features=features)
    return other
  def to(self, device, copy: bool = False):
    if not copy and self.device == device: return self
    other = self.clone()
    if self.device == device: return other
    raise NotImplementedError
  def cpu(self): return self.to(torch.device("cpu"))
  def cuda(self): return self.to(torch.device("cuda"))
  def isempty(self) -> bool: return self._N == 0 or self.valid.eq(False).all()

