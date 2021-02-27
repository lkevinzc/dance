from typing import List, Union, Tuple
import torch
from detectron2.structures import Boxes

from detectron2.layers import cat


class ExtremePoints:
    def __init__(self, tensor: torch.Tensor):
        """
        :param tensor (Tensor[float]): a Nx4x2 tensor.  Last dim is (x, y); second last follows [tt, ll, bb, rr]:
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 4, 2, dtype=torch.float32, device=device)
        assert tensor.dim() == 3 and tensor.size(-1) == 2, tensor.size()

        self.tensor = tensor

        self.spanned_nodes = []

        self.box = None

    def clone(self) -> "ExtremePoints":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return ExtremePoints(self.tensor.clone())

    def to(self, device: str) -> "ExtremePoints":
        return ExtremePoints(self.tensor.to(device))

    def get_boxes(self) -> Boxes:
        bboxes = torch.stack([
            self.tensor[:, 1, 0],
            self.tensor[:, 0, 1],
            self.tensor[:, 3, 0],
            self.tensor[:, 2, 1],
        ], dim=1)
        return Boxes(bboxes)

    def compute_on_ext_centered_masks(self, N, edge_map, radius, mode, image_shape):
        self.spread(N, radius, mode, image_shape)
        m = torch.zeros((N,) + image_shape, device=self.device)
        num_nodes = []
        for i, node in enumerate(self.spanned_nodes):
            node = node.long()
            m[i, node[:, 1], node[:, 0]] = 1    # TODO: trigger CUDA assert, ...?
            num_nodes.append(node.size(0))

        edge_map_m = edge_map.unsqueeze(0) * m
        instance_score = edge_map_m.sum(dim=1).sum(dim=1)
        num_nodes = torch.tensor(num_nodes, device=self.device)
        return instance_score / num_nodes

    def compute_by_grid_sample(self, N, edge_map, radius, mode, image_shape):
        self.spread(N, radius, mode, image_shape)
        mean_scores = []
        for i, node in enumerate(self.spanned_nodes):
            sampled_nodes = torch.nn.functional.grid_sample(edge_map.unsqueeze(0).unsqueeze(0),
                                                            node.unsqueeze(0).unsqueeze(0))
            mean_scores.append(sampled_nodes.mean())
        return torch.stack(mean_scores)

    def spread(self, N, radius, mode, image_shape):
        """
        Spreads the extreme points for robustness.
        :param N: (int) number of instances
        :param radius:  (int) circle radius
        :param mode: (str) 'linear' or 'gaussian', # TODO now only support linear
        """
        if len(self.spanned_nodes) == N:
            # avoid re-compute
            return
        assert mode == 'linear', 'unsupported mode'
        h, w = image_shape
        box = self.get_boxes().tensor
        whs = torch.stack([(box[:, 2] - box[:, 0]), (box[:, 3] - box[:, 1])], dim=1)
        num_pix_r = (whs * radius).floor()
        ext_pts = self.tensor
        for i in range(N):
            per_num_pix_r = num_pix_r[i]
            per_ext_pts = ext_pts[i]
            square_area = int((per_num_pix_r[0] * 2 + 1) * (per_num_pix_r[1] * 2 + 1))
            per_spanned_pts = per_ext_pts.repeat_interleave(int(square_area), dim=0)
            span_xs = torch.arange(-int(per_num_pix_r[0]), int(per_num_pix_r[0]) + 1,
                                   step=1, dtype=torch.float32, device=self.device)
            span_ys = torch.arange(-int(per_num_pix_r[1]), int(per_num_pix_r[1]) + 1,
                                   step=1, dtype=torch.float32, device=self.device)
            span_y, span_x = torch.meshgrid(span_ys, span_xs)
            span_xy = torch.stack([span_x.reshape(-1), span_y.reshape(-1)], dim=1)
            # (4 * square_area, 2)
            per_spanned_nodes = (per_spanned_pts + span_xy.repeat(4,1)).floor()
            per_spanned_nodes[:, 0].clamp_(min=0, max=w-1)
            per_spanned_nodes[:, 1].clamp_(min=0, max=h-1)
            self.spanned_nodes.append(per_spanned_nodes)

    # TODO (ZC): rename
    def onedge(self,
             edge_map: torch.Tensor,
             image_shape,
             radius=1/20,
             threshold=0.1,
             compute_method='masking') -> torch.Tensor:
        """
        This forces every instance to "think" semantically and globally about its existence.
        :param edge_map: (torch.Tensor) HxW (image_shape), be aware of the scaling
        :param image_shape: (tuple of 2) desired output shape
        :param radius: (float) controls how big the circle is spanned
        :param threshold: (float) to suppress instances
        :param compute_method: (str) specifies computing method (speed testing)
        :return: (torch.Tensor) a binary vector which represents whether each instance is on edge
                (False) or not (True).
        """
        ext_pts = self.tensor  # N x 4 x 2
        N = len(ext_pts)
        if N == 0:
            # nothing to quit
            return torch.tensor(0)
        if compute_method == 'masking':
            instance_score = self.compute_on_ext_centered_masks(N, edge_map, radius, 'linear', image_shape)
            keep = instance_score > threshold
            return keep
        elif compute_method == 'sampling':
            instance_score = self.compute_by_grid_sample(N, edge_map, radius, 'linear', image_shape)
            keep = instance_score > threshold
            return keep
        else:
            raise ValueError('Unsupported compute method:', compute_method)

    def align(self, pooler_resolution):
        box = self.get_boxes().tensor
        w = box[:, 2] - box[:, 0] + 1
        h = box[:, 3] - box[:, 1] + 1
        de_location = self.tensor - box[:, None, :2]
        de_location[:, :, 0] /= w[:, None] / pooler_resolution  # x
        de_location[:, :, 1] /= h[:, None] / pooler_resolution  # y
        return de_location.int()

    @staticmethod
    def from_boxes(boxes: Boxes, offsets: torch.Tensor, locations: torch.Tensor) -> "ExtremePoints":
        """
        Generate the ExtremePoints from a box and offset along each edge, with locations bing origins;
        the outputs will correspond to the input boxes
        :param boxes (Boxes): from Nx4 tensor matrix.
        :param offsets (torch.Tensor): float matrix of Nx4.
        :param locations (torch.Tensor): float matrix of Nx2, indicating corresponding locations
        :return: ExtremePoints
        """
        x1 = boxes.tensor[:, 0] # ll_x
        y1 = boxes.tensor[:, 1] # tt_y
        x2 = boxes.tensor[:, 2] # rr_x
        y2 = boxes.tensor[:, 3] # bb_y
        w = x2 - x1
        h = y2 - y1
        tt_x = (locations[:, 0] + w * offsets[:, 0])
        ll_y = (locations[:, 1] + h * offsets[:, 1])
        bb_x = (locations[:, 0] + w * offsets[:, 2])
        rr_y = (locations[:, 1] + h * offsets[:, 3])

        return ExtremePoints(torch.stack([tt_x, y1, x1, ll_y, bb_x, y2, x2, rr_y], dim=1).view(-1, 4, 2))

    def fit_to_box(self):
        box = self.get_boxes().tensor
        n = box.size(0)
        lower_bound = box.view(-1, 2, 2)[:, :1, :]
        upper_bound = box.view(-1, 2, 2)[:, 1:, :]
        beyond_lower = self.tensor < lower_bound
        beyond_upper = self.tensor > upper_bound
        if beyond_lower.any():
            self.tensor[beyond_lower] = lower_bound.expand(n, 4, 2)[beyond_lower]
        if beyond_upper.any():
            self.tensor[beyond_upper] = upper_bound.expand(n, 4, 2)[beyond_upper]

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, :, 0] *= scale_x
        self.tensor[:, :, 1] *= scale_y

    def get_octagons(self, frac=8.):
        # counter clock wise
        ext_pts = self.tensor   # N x 4 x 2
        N = len(ext_pts)
        if N == 0:
            return ext_pts.new_empty(0, 16)
        w, h = ext_pts[:, 3, 0] - ext_pts[:, 1, 0], ext_pts[:, 2, 1] - ext_pts[:, 0, 1]
        t, l, b, r = ext_pts[:, 0, 1], ext_pts[:, 1, 0], ext_pts[:, 2, 1], ext_pts[:, 3, 0]
        x1, y1 = torch.min(ext_pts[:, 0, 0] + w / frac, r), ext_pts[:, 0, 1]
        x2, y2 = torch.max(ext_pts[:, 0, 0] - w / frac, l), ext_pts[:, 0, 1]
        x3, y3 = ext_pts[:, 1, 0], torch.max(ext_pts[:, 1, 1] - h / frac, t)
        x4, y4 = ext_pts[:, 1, 0], torch.min(ext_pts[:, 1, 1] + h / frac, b)
        x5, y5 = torch.max(ext_pts[:, 2, 0] - w / frac, l), ext_pts[:, 2, 1]
        x6, y6 = torch.min(ext_pts[:, 2, 0] + w / frac, r), ext_pts[:, 2, 1]
        x7, y7 = ext_pts[:, 3, 0], torch.min(ext_pts[:, 3, 1] + h / frac, b)
        x8, y8 = ext_pts[:, 3, 0], torch.max(ext_pts[:, 3, 1] - h / frac, t)
        octagons = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4,
                               x5, y5, x6, y6, x7, y7, x8, y8], dim=1)
        return octagons

    def area(self) -> torch.Tensor:
        return self.get_boxes().area()

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "ExtremePoints":
        """
        Returns:
            ExtremePoints: Create a new :class:`ExtremePoints` by indexing.

        The following usage are allowed:

        1. `new_exts = exts[3]`: return a `ExtremePoints` which contains only one box.
        2. `new_exts = exts[2:10]`: return a slice of extreme points.
        3. `new_exts = exts[vector]`, where vector is a torch.BoolTensor
           with `length = len(exts)`. Nonzero elements in the vector will be selected.

        Note that the returned ExtremePoints might share storage with this ExtremePoints,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return ExtremePoints(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 3, "Indexing on ExtremePoints with {} failed to return a matrix!".format(item)
        return ExtremePoints(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "ExtPts(" + str(self.tensor) + ")"

    @staticmethod
    def cat(pts_list: List["ExtremePoints"]) -> "ExtremePoints":
        """
        Concatenates a list of ExtremePoints into a single ExtremePoints

        Arguments:
            pts_list (list[ExtremePoints])

        Returns:
            pts: the concatenated ExtremePoints
        """
        assert isinstance(pts_list, (list, tuple))
        assert len(pts_list) > 0
        assert all(isinstance(pts, ExtremePoints) for pts in pts_list)

        cat_pts = type(pts_list[0])(cat([p.tensor for p in pts_list], dim=0))
        return cat_pts

    @property
    def device(self) -> torch.device:
        return self.tensor.device


class PolygonPoints:
    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        """
        :param tensor (Tensor[float]): a Nxkx2 tensor.  Last dim is (x, y);
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 128, 2, dtype=torch.float32, device=device)
        assert tensor.dim() == 3 and tensor.size(-1) == 2, tensor.size()

        self.tensor = tensor

    def clone(self) -> "PolygonPoints":

        return PolygonPoints(self.tensor.clone())

    def to(self, device: str) -> "PolygonPoints":
        return PolygonPoints(self.tensor.to(device))

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, :, 0] *= scale_x
        self.tensor[:, :, 1] *= scale_y

    def clip(self, box_size: BoxSizeType) -> None:
        assert torch.isfinite(self.tensor).all(), "Polygon tensor contains infinite or NaN!"
        h, w = box_size
        self.tensor[:, :, 0].clamp_(min=0, max=w)
        self.tensor[:, :, 1].clamp_(min=0, max=h)

    def flatten(self):
        n = self.tensor.size(0)
        if n == 0:
            return self.tensor
        return self.tensor.reshape(n, -1)

    def get_box(self):
        return torch.cat([self.tensor.min(1)[0], self.tensor.max(1)[0]], dim=1)

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "PolygonPoints":
        """
        Returns:
            ExtremePoints: Create a new :class:`ExtremePoints` by indexing.

        The following usage are allowed:

        1. `new_exts = exts[3]`: return a `ExtremePoints` which contains only one box.
        2. `new_exts = exts[2:10]`: return a slice of extreme points.
        3. `new_exts = exts[vector]`, where vector is a torch.BoolTensor
           with `length = len(exts)`. Nonzero elements in the vector will be selected.

        Note that the returned ExtremePoints might share storage with this ExtremePoints,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return PolygonPoints(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 3, "Indexing on PolygonPoints with {} failed to return a matrix!".format(item)
        return PolygonPoints(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "PolyPts(" + str(self.tensor) + ")"

    @staticmethod
    def cat(pts_list: List["PolygonPoints"]) -> "PolygonPoints":
        """
        Concatenates a list of ExtremePoints into a single ExtremePoints

        Arguments:
            pts_list (list[PolygonPoints])

        Returns:
            pts: the concatenated PolygonPoints
        """
        assert isinstance(pts_list, (list, tuple))
        assert len(pts_list) > 0
        assert all(isinstance(pts, PolygonPoints) for pts in pts_list)

        cat_pts = type(pts_list[0])(cat([p.tensor for p in pts_list], dim=0))
        return cat_pts

    @property
    def device(self) -> torch.device:
        return self.tensor.device


