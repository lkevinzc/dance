import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
from shapely.geometry import Polygon

from detectron2.layers import Conv2d, cat, ShapeSpec
from core.structures import ExtremePoints, PolygonPoints

from core.layers import DFConv2d, SmoothL1Loss, extreme_utils

from core.modeling.fcose.utils import get_extreme_points


class DilatedCircularConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircularConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(
            state_dim,
            out_state_dim,
            kernel_size=self.n_adj * 2 + 1,
            dilation=self.dilation,
        )

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat(
                [
                    input[..., -self.n_adj * self.dilation :],
                    input,
                    input[..., : self.n_adj * self.dilation],
                ],
                dim=2,
            )
        return self.fc(input)


class GraphConvolution(nn.Module):
    """ http://snap.stanford.edu/proj/embeddings-www/files/nrltutorial-part2-gnns.pdf p.19 """

    def __init__(self, state_dim, out_state_dim=None):
        super(GraphConvolution, self).__init__()

        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc1 = nn.Conv1d(state_dim, out_state_dim, 1)
        self.fc2 = nn.Conv1d(state_dim, out_state_dim, 1)

    def forward(self, input, adj):
        state_in = self.fc1(input)
        forward_input = (
            input[..., adj.view(-1)]
            .view(input.size(0), -1, adj.size(0), adj.size(1))
            .mean(3)
        )
        forward_input = self.fc2(forward_input)
        return state_in + forward_input


class SnakeBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, n_adj=4, dilation=1):
        super(SnakeBlock, self).__init__()

        self.conv = DilatedCircularConv(state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim):
        super(BasicBlock, self).__init__()

        self.conv = GraphConvolution(state_dim, out_state_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x, adj):
        x = self.conv(x, adj)
        x = self.relu(x)
        x = self.norm(x)
        return x


class _SnakeNet(nn.Module):
    def __init__(self, state_dim, feature_dim):
        super(_SnakeNet, self).__init__()

        self.head = SnakeBlock(feature_dim, state_dim)

        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        for i in range(self.res_layer_num):
            conv = SnakeBlock(state_dim, state_dim, n_adj=4, dilation=dilation[i])
            self.__setattr__("res" + str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(
            state_dim * (self.res_layer_num + 1), fusion_state_dim, 1
        )
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1),
        )

    def forward(self, x):
        states = []

        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__("res" + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(
            global_state.size(0), global_state.size(1), state.size(2)
        )
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x


class _ResGCNNet(nn.Module):
    def __init__(self, state_dim, feature_dim):
        super(_ResGCNNet, self).__init__()

        self.head = BasicBlock(feature_dim, state_dim)

        self.res_layer_num = 7
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim)
            self.__setattr__("res" + str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(
            state_dim * (self.res_layer_num + 1), fusion_state_dim, 1
        )
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1),
        )

    def forward(self, x, adj):
        states = []

        x = self.head(x, adj)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__("res" + str(i))(x, adj) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(
            global_state.size(0), global_state.size(1), state.size(2)
        )
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x


class SnakeFPNHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # serves like a dummy module
        self.refine_head = DsnakeHead(cfg)

    def forward(self, features, pred_instances=None, targets=None):
        if self.training:
            _, losses = self.refine_head(features["p2"], None, targets)
            return losses, []
        else:
            new_instances, _ = self.refine_head(features["p2"], pred_instances, None)
            return {}, new_instances


class DsnakeHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        conv_dims = cfg.MODEL.EDGE_HEAD.CONVS_DIM
        norm = cfg.MODEL.EDGE_HEAD.NORM

        # Snake settings
        self.conv_type = cfg.MODEL.SNAKE_HEAD.CONV_TYPE
        self.initial = cfg.MODEL.SNAKE_HEAD.INITIAL
        self.num_adj = cfg.MODEL.SNAKE_HEAD.FILTER_WIDTH
        self.num_iter = cfg.MODEL.SNAKE_HEAD.NUM_ITER
        self.num_convs = cfg.MODEL.SNAKE_HEAD.NUM_CONVS
        self.num_sampling = cfg.MODEL.SNAKE_HEAD.NUM_SAMPLING
        self.de_location_type = cfg.MODEL.SNAKE_HEAD.DE_LOC_TYPE
        self.dilations = cfg.MODEL.SNAKE_HEAD.DILATIONS
        self.reorder_method = cfg.MODEL.SNAKE_HEAD.REORDER_METHOD

        self.visualize_path = cfg.MODEL.SNAKE_HEAD.VIS_PATH

        self.refine_loss_weight = 10

        if cfg.MODEL.SNAKE_HEAD.LOSS_TYPE == "smoothl1":
            self.loss_reg = SmoothL1Loss(beta=cfg.MODEL.SNAKE_HEAD.LOSS_L1_BETA)
        else:
            raise ValueError("loss undefined!")

        self.bottom_out = nn.ModuleList()
        for i in range(self.num_convs):

            norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
            conv = Conv2d(
                conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=self.dilations[i],
                bias=not norm,
                norm=norm_module,
                activation=F.relu,
            )
            # weight_init.c2_msra_fill(conv) - DONT USE THIS INIT
            self.bottom_out.append(conv)

        self.fuse = nn.Conv1d(2 * conv_dims, conv_dims, 1)

        self.init_snake = _SnakeNet(state_dim=128, feature_dim=conv_dims + 2)

        for i in range(len(self.num_iter)):
            if self.conv_type == "ccn":
                snake_deformer = _SnakeNet(state_dim=128, feature_dim=conv_dims + 2)
            elif self.conv_type == "gcn":
                snake_deformer = _ResGCNNet(state_dim=128, feature_dim=conv_dims + 2)
            else:
                raise ValueError("Unsupported operation!")
            self.__setattr__("deformer" + str(i), snake_deformer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_simple_extreme_points(polygons_lists):
        ext_pts = torch.zeros([len(polygons_lists), 4])
        for i, ins_polys in enumerate(polygons_lists):
            if len(ins_polys):
                flatten_polys = ins_polys[0]
            else:
                flatten_polys = np.concatenate(ins_polys)

            flatten_polys = flatten_polys.reshape(-1, 2)
            ext_pt = get_extreme_points(flatten_polys)
            ext_pts[i] = torch.from_numpy(ext_pt)
        return ext_pts

    @staticmethod
    def uniform_sample(pgtnp_px2, newpnum):
        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths
        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum :]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            for i in range(pnum):
                if edgenum[i] == 0:
                    edgenum[i] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:

                if edgenumsum > newpnum:

                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i : i + 1]
                pe_1x2 = pgtnext_px2[i : i + 1]

                pnewnum = edgenum[i]
                wnp_kx1 = (
                    np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]
                )

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp

    @staticmethod
    def uniform_upsample(poly, p_num):
        if poly.size(1) == 0:
            return torch.zeros([poly.size(0), 0, p_num, 2]).to(poly.device)

        # 1. assign point number for each edge
        # 2. calculate the coefficient for linear interpolation
        next_poly = torch.roll(poly, -1, 2)
        edge_len = (next_poly - poly).pow(2).sum(3).sqrt()
        edge_num = torch.round(
            edge_len * p_num / torch.sum(edge_len, dim=2)[..., None]
        ).long()
        edge_num = torch.clamp(edge_num, min=1)
        edge_num_sum = torch.sum(edge_num, dim=2)
        edge_idx_sort = torch.argsort(edge_num, dim=2, descending=True)
        extreme_utils.calculate_edge_num(edge_num, edge_num_sum, edge_idx_sort, p_num)
        edge_num_sum = torch.sum(edge_num, dim=2)
        assert torch.all(edge_num_sum == p_num)

        edge_start_idx = torch.cumsum(edge_num, dim=2) - edge_num
        weight, ind = extreme_utils.calculate_wnp(edge_num, edge_start_idx, p_num)
        poly1 = poly.gather(
            2, ind[..., 0:1].expand(ind.size(0), ind.size(1), ind.size(2), 2)
        )
        poly2 = poly.gather(
            2, ind[..., 1:2].expand(ind.size(0), ind.size(1), ind.size(2), 2)
        )
        poly = poly1 * (1 - weight) + poly2 * weight

        return poly[0]

    def get_octagon(self, ex, p_num):

        if len(ex) == 0:
            return torch.zeros([0, p_num, 2]).to(ex.device)

        ex = ex[None]
        w, h = ex[..., 3, 0] - ex[..., 1, 0], ex[..., 2, 1] - ex[..., 0, 1]
        t, l, b, r = ex[..., 0, 1], ex[..., 1, 0], ex[..., 2, 1], ex[..., 3, 0]
        x = 8.0

        octagon = [
            ex[..., 0, 0],
            ex[..., 0, 1],
            torch.max(ex[..., 0, 0] - w / x, l),
            ex[..., 0, 1],
            ex[..., 1, 0],
            torch.max(ex[..., 1, 1] - h / x, t),
            ex[..., 1, 0],
            ex[..., 1, 1],
            ex[..., 1, 0],
            torch.min(ex[..., 1, 1] + h / x, b),
            torch.max(ex[..., 2, 0] - w / x, l),
            ex[..., 2, 1],
            ex[..., 2, 0],
            ex[..., 2, 1],
            torch.min(ex[..., 2, 0] + w / x, r),
            ex[..., 2, 1],
            ex[..., 3, 0],
            torch.min(ex[..., 3, 1] + h / x, b),
            ex[..., 3, 0],
            ex[..., 3, 1],
            ex[..., 3, 0],
            torch.max(ex[..., 3, 1] - h / x, t),
            torch.min(ex[..., 0, 0] + w / x, r),
            ex[..., 0, 1],
        ]
        octagon = torch.stack(octagon, dim=2).view(t.size(0), t.size(1), 12, 2)
        octagon = self.uniform_upsample(octagon, p_num)
        return octagon

    @staticmethod
    def dim0_roll(x, n):
        """
        single dimension roll
        :param x:
        :param n:
        :return:
        """
        return torch.cat((x[n:, ...], x[..., :n]))

    def sample_octagons_fast(self, pred_instances):
        poly_sample_locations = []
        image_index = []
        for im_i in range(len(pred_instances)):
            instance_per_im = pred_instances[im_i]
            ext_points = instance_per_im.ext_points.tensor  # (n, 4, 2)
            poly_sample_locations.append(
                self.get_octagon(ext_points, self.num_sampling)
            )
            image_index.append(ext_points.new_empty(len(ext_points)).fill_(im_i))

        poly_sample_locations = cat(poly_sample_locations, dim=0)
        image_index = cat(image_index)
        return poly_sample_locations, image_index

    def sample_quadrangles_fast(self, pred_instances):
        init_sample_locations = []
        image_index = []
        for im_i in range(len(pred_instances)):
            instance_per_im = pred_instances[im_i]
            # (num_det, 4, 2)
            quad = self.get_quadrangle(instance_per_im.pred_boxes.tensor).view(-1, 4, 2)
            # (num_det, 40, 2)
            init_sample_locations.append(self.uniform_upsample(quad[None], 40))
            image_index.append(quad.new_empty(len(quad)).fill_(im_i))

        init_sample_locations = cat(init_sample_locations, dim=0)
        image_index = cat(image_index)
        return init_sample_locations, image_index

    @staticmethod
    def get_simple_contour(gt_masks):
        polygon_mask = gt_masks.polygons
        contours = []

        for polys in polygon_mask:
            # polys = binary_mask_to_polygon(mask)
            contour = list(
                map(lambda x: np.array(x).reshape(-1, 2).astype(np.float32), polys)
            )
            if len(contour) > 1:  # ignore fragmented instances
                contours.append(None)
            else:
                contours.append(contour[0])
        return contours

    def sample_octagons(self, pred_instances):
        poly_sample_locations = []
        image_index = []
        for im_i in range(len(pred_instances)):
            instance_per_im = pred_instances[im_i]
            ext_points = instance_per_im.ext_points
            octagons_per_im = ext_points.get_octagons().cpu().numpy().reshape(-1, 8, 2)
            for oct in octagons_per_im:
                # sampling from octagon
                oct_sampled_pts = self.uniform_sample(oct, self.num_sampling)

                oct_sampled_pts = (
                    oct_sampled_pts[::-1]
                    if Polygon(oct_sampled_pts).exterior.is_ccw
                    else oct_sampled_pts
                )
                assert not Polygon(
                    oct_sampled_pts
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                poly_sample_locations.append(
                    torch.tensor(oct_sampled_pts, device=ext_points.device)
                )
                image_index.append(im_i)

        if not poly_sample_locations:
            return poly_sample_locations, image_index

        poly_sample_locations = torch.stack(poly_sample_locations, dim=0)
        image_index = torch.tensor(image_index)
        return poly_sample_locations, image_index

    @staticmethod
    def get_quadrangle(box):
        if len(box) == 0:
            return torch.zeros([0, 4, 2]).to(box.device)
        x_min, y_min, x_max, y_max = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        quadrangle = [
            (x_min + x_max) / 2.0,
            y_min,
            x_min,
            (y_min + y_max) / 2.0,
            (x_min + x_max) / 2.0,
            y_max,
            x_max,
            (y_min + y_max) / 2.0,
        ]
        quadrangle = torch.stack(quadrangle, dim=1)
        return quadrangle

    def compute_targets_for_polys(self, targets):
        init_sample_locations = []
        init_sample_targets = []
        poly_sample_locations = []
        poly_sample_targets = []
        image_index = []
        scales = []

        # per image
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor

            # no gt
            if bboxes.numel() == 0:
                continue

            gt_masks = targets_per_im.gt_masks

            # use this as a scaling
            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]

            quadrangle = (
                self.get_quadrangle(bboxes).cpu().numpy().reshape(-1, 4, 2)
            )  # (k, 4, 2)

            if self.initial == "octagon":
                # [t_H_off, l_V_off, b_H_off, r_V_off]
                ext_pts_off = self.get_simple_extreme_points(gt_masks.polygons).to(
                    bboxes.device
                )
                ex_t = torch.stack([ext_pts_off[:, None, 0], bboxes[:, None, 1]], dim=2)
                ex_l = torch.stack([bboxes[:, None, 0], ext_pts_off[:, None, 1]], dim=2)
                ex_b = torch.stack([ext_pts_off[:, None, 2], bboxes[:, None, 3]], dim=2)
                ex_r = torch.stack([bboxes[:, None, 2], ext_pts_off[:, None, 3]], dim=2)

                # k x 4 x 2
                ext_points = torch.cat([ex_t, ex_l, ex_b, ex_r], dim=1)

                # N x 16 (ccw)
                octagons = (
                    ExtremePoints(torch.cat([ex_t, ex_l, ex_b, ex_r], dim=1))
                    .get_octagons()
                    .cpu()
                    .numpy()
                    .reshape(-1, 8, 2)
                )
            else:
                raise ValueError("Invalid initial input!")

            # List[nd.array], element shape: (P, 2) OR None
            contours = self.get_simple_contour(gt_masks)

            # per instance
            for (quad, oct, cnt, ext, w, h) in zip(
                quadrangle, octagons, contours, ext_points, ws, hs
            ):
                if cnt is None:
                    continue

                # used for normalization
                scale = torch.min(w, h)

                # make it clock-wise
                cnt = cnt[::-1] if Polygon(cnt).exterior.is_ccw else cnt
                assert not Polygon(
                    cnt
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                # sampling from quadrangle
                # print(quad.shape)
                # print(oct.shape)
                quad_sampled_pts = self.uniform_sample(quad, 40)

                # sampling from octagon
                oct_sampled_pts = self.uniform_sample(oct, self.num_sampling)

                oct_sampled_pts = (
                    oct_sampled_pts[::-1]
                    if Polygon(oct_sampled_pts).exterior.is_ccw
                    else oct_sampled_pts
                )
                assert not Polygon(
                    oct_sampled_pts
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                # sampling from ground truth
                oct_sampled_targets = self.uniform_sample(
                    cnt, len(cnt) * self.num_sampling
                )  # (big, 2)
                # i) find a single nearest, so that becomes ordered point sets

                tt_idx = np.argmin(
                    np.power(oct_sampled_targets - oct_sampled_pts[0], 2).sum(axis=1)
                )
                oct_sampled_targets = np.roll(oct_sampled_targets, -tt_idx, axis=0)[
                    :: len(cnt)
                ]

                # assert not Polygon(oct_sampled_targets).exterior.is_ccw, '2) contour must be clock-wise!'

                quad_sampled_pts = torch.tensor(quad_sampled_pts, device=bboxes.device)
                oct_sampled_pts = torch.tensor(oct_sampled_pts, device=bboxes.device)
                oct_sampled_targets = torch.tensor(
                    oct_sampled_targets, device=bboxes.device
                )

                # oct_sampled_targets = gt_sampled_pts - oct_sampled_pts  # offset field

                init_sample_locations.append(quad_sampled_pts)
                init_sample_targets.append(ext)
                poly_sample_locations.append(oct_sampled_pts)
                poly_sample_targets.append(oct_sampled_targets)
                image_index.append(im_i)
                scales.append(scale)

        init_sample_locations = torch.stack(init_sample_locations, dim=0)
        init_sample_targets = torch.stack(init_sample_targets, dim=0)
        poly_sample_locations = torch.stack(poly_sample_locations, dim=0)
        poly_sample_targets = torch.stack(poly_sample_targets, dim=0)
        image_index = torch.tensor(image_index, device=bboxes.device)
        scales = torch.stack(scales, dim=0)
        return {
            "quadrangle_locs": init_sample_locations,
            "quadrangle_targets": init_sample_targets,
            "octagon_locs": poly_sample_locations,
            "octagon_targets": poly_sample_targets,
            "scales": scales,
            "image_idx": image_index,
        }

    def reorder(self, oct_sampled_targets, oct_sampled_pts):
        """
        :param oct_sampled_targets: N x 2 for single instance
        :param oct_sampled_pts:
        :return:
        """
        ind1 = torch.arange(self.num_sampling, device=oct_sampled_targets.device)
        ind2 = ind1.expand(self.num_sampling, -1)
        enumerated_ind = (
            torch.fmod(ind2 + ind1.view(-1, 1), self.num_sampling).view(-1).long()
        )
        enumerated_targets = oct_sampled_targets[enumerated_ind].reshape(
            -1, self.num_sampling, 2
        )
        diffs = enumerated_targets - oct_sampled_pts
        tt_idx = torch.argmin(diffs.pow(2).sum(2).sum(1))
        # print(tt_idx)
        return enumerated_targets[tt_idx]

    def get_locations_feature(self, features, locations, image_idx):
        """
        :param feat: list like [(b, c, h/s, w/s), ...]
        :param img_poly: (Sigma{num_poly_i}, poly_num, 2) - scaled by s
        :param ind: poly corresponding index to image
        :param lvl: list, poly corresponding index to feat level
        :return:
        """
        h = features.shape[2] * 4
        w = features.shape[3] * 4
        locations = locations.clone()
        locations[..., 0] = locations[..., 0] / (w / 2.0) - 1
        locations[..., 1] = locations[..., 1] / (h / 2.0) - 1

        batch_size = features.size(0)
        sampled_features = torch.zeros(
            [locations.size(0), features.size(1), locations.size(1)]
        ).to(locations.device)
        for i in range(batch_size):
            per_im_loc = locations[image_idx == i].unsqueeze(0)
            feature = torch.nn.functional.grid_sample(
                features[i : i + 1], per_im_loc, align_corners=False
            )[0].permute(1, 0, 2)
            sampled_features[image_idx == i] = feature

        return sampled_features

    def de_location(self, locations):
        # de-location (spatial relationship among locations; translation invariant)
        x_min = torch.min(locations[..., 0], dim=-1)[0]
        y_min = torch.min(locations[..., 1], dim=-1)[0]
        # x_max = torch.max(locations[..., 0], dim=-1)[0]
        # y_max = torch.max(locations[..., 1], dim=-1)[0]
        new_locations = locations.clone()

        # TODO (Zc): no normalization, this helps maitain the shape I think~
        new_locations[..., 0] = new_locations[..., 0] - x_min[..., None]
        new_locations[..., 1] = new_locations[..., 1] - y_min[..., None]
        return new_locations

    @staticmethod
    def get_adj_ind(n_adj, n_nodes, device):
        ind = torch.LongTensor(
            [i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0]
        )
        ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
        return ind.to(device)

    def init(self, deformer, features, locations, image_idx):
        sampled_features = self.get_locations_feature(features, locations, image_idx)
        # (\sum{k}, 2)
        # TODO: same as dsnake, use center to get more context
        center = (torch.min(locations, dim=1)[0] + torch.max(locations, dim=1)[0]) * 0.5
        center_feat = self.get_locations_feature(features, center[:, None], image_idx)

        # (Sigma{num_poly_i}, 2 * feat_dim, 128)
        init_feat = torch.cat(
            [sampled_features, center_feat.expand_as(sampled_features)], dim=1
        )
        init_feat = self.fuse(init_feat)
        calibrated_locations = self.de_location(locations)
        concat_features = torch.cat(
            [init_feat, calibrated_locations.permute(0, 2, 1)], dim=1
        )

        if self.conv_type == "ccn":
            pred_offsets = deformer(concat_features).permute(0, 2, 1)
        elif self.conv_type == "gcn":
            adj = self.get_adj_ind(
                self.num_adj, concat_features.size(2), concat_features.device
            )
            pred_offsets = deformer(concat_features, adj).permute(0, 2, 1)
        else:
            raise ValueError("Unsupported operation!")
        pred_locations = locations + pred_offsets
        pred_exts = pred_locations[:, ::10]  # only take the 4 corners
        return pred_exts

    def evolve(self, deformer, features, locations, image_idx):
        sampled_features = self.get_locations_feature(features, locations, image_idx)
        calibrated_locations = self.de_location(locations)
        # (Sigma{num_poly_i}, 2 + feat_dim, 128)
        concat_features = torch.cat(
            [sampled_features, calibrated_locations.permute(0, 2, 1)], dim=1
        )
        if self.conv_type == "ccn":
            pred_offsets = deformer(concat_features).permute(0, 2, 1)
        elif self.conv_type == "gcn":
            adj = self.get_adj_ind(
                self.num_adj, concat_features.size(2), concat_features.device
            )
            pred_offsets = deformer(concat_features, adj).permute(0, 2, 1)
        else:
            raise ValueError("Unsupported operation!")
        pred_locations = locations + pred_offsets
        return pred_locations

    def forward(self, features, pred_instances=None, targets=None):
        if self.training:
            training_targets = self.compute_targets_for_polys(targets)
            locations, reg_targets, scales, image_idx = (
                training_targets["octagon_locs"],
                training_targets["octagon_targets"],
                training_targets["scales"],
                training_targets["image_idx"],
            )
            init_locations, init_targets = (
                training_targets["quadrangle_locs"],
                training_targets["quadrangle_targets"],
            )

        else:
            assert pred_instances is not None
            init_locations, image_idx = self.sample_quadrangles_fast(pred_instances)
            if len(init_locations) == 0:
                return pred_instances, {}

        # enhance bottom features TODO: maybe reduce later
        for i in range(self.num_convs):
            features = self.bottom_out[i](features)

        pred_exts = self.init(self.init_snake, features, init_locations, image_idx)

        if not self.training:
            h = features.shape[2] * 4
            w = features.shape[3] * 4

            poly_sample_locations = []
            for i, instance_per_im in enumerate(pred_instances):
                pred_exts_per_im = pred_exts[image_idx == i]  # N x 4 x 2
                pred_exts_per_im[..., 0] = torch.clamp(
                    pred_exts_per_im[..., 0], min=0, max=w - 1
                )
                pred_exts_per_im[..., 1] = torch.clamp(
                    pred_exts_per_im[..., 1], min=0, max=h - 1
                )
                if not instance_per_im.has("ext_points"):
                    instance_per_im.ext_points = ExtremePoints(pred_exts_per_im)
                    poly_sample_locations.append(
                        self.get_octagon(pred_exts_per_im, self.num_sampling)
                    )
                else:  # NOTE: For GT Input testing
                    # print('Using GT EX')
                    poly_sample_locations.append(
                        self.get_octagon(
                            instance_per_im.ext_points.tensor, self.num_sampling
                        )
                    )
            locations = cat(poly_sample_locations, dim=0)

        location_preds = []

        for i in range(len(self.num_iter)):
            deformer = self.__getattr__("deformer" + str(i))
            if i == 0:
                pred_location = self.evolve(deformer, features, locations, image_idx)
            else:
                pred_location = self.evolve(
                    deformer, features, pred_location, image_idx
                )
            location_preds.append(pred_location)

        if self.training:
            evolve_loss = 0
            for pred in location_preds:
                evolve_loss += (
                    self.loss_reg(
                        pred / scales[:, None, None],
                        reg_targets / scales[:, None, None],
                    )
                    / 3
                )

            init_loss = self.loss_reg(
                pred_exts / scales[:, None, None], init_targets / scales[:, None, None]
            )
            losses = {
                "loss_evolve": evolve_loss * self.refine_loss_weight,
                "loss_init": init_loss * self.refine_loss_weight,
            }
            return [], losses
        else:
            new_instances = self.predict_postprocess(
                pred_instances, locations, location_preds, image_idx
            )
            return new_instances, {}

    def predict_postprocess(self, pred_instances, locations, location_preds, image_idx):
        results = []
        # per im
        for i, instance_per_im in enumerate(pred_instances):
            pred_per_im = location_preds[-1][image_idx == i]  # N x 128 x 2
            instance_per_im.pred_polys = PolygonPoints(pred_per_im)

            if self.visualize_path:
                path = []
                loc_per_im = locations[image_idx == i]
                path.append(loc_per_im)

                for k in range(len(self.num_iter) - 1):
                    nodes_per_im = location_preds[k][image_idx == i]
                    path.append(nodes_per_im)

                path = torch.stack(path, dim=1)
                instance_per_im.pred_path = path

            results.append(instance_per_im)
        return results


def __vis(image, poly_sample_locations, poly_sample_targets):
    import matplotlib.pyplot as plt

    image = image.cpu().numpy().transpose(1, 2, 0)[:, :, ::-1].astype(np.uint8)
    poly_sample_locations = poly_sample_locations.cpu().numpy()
    poly_sample_targets = poly_sample_targets.cpu().numpy()
    colors = (
        np.array([[1, 1, 198], [51, 1, 148], [101, 1, 98], [151, 1, 48], [201, 1, 8]])
        / 255.0
    )

    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.tight_layout()

    ax.imshow(image)

    for i, (loc, target) in enumerate(zip(poly_sample_locations, poly_sample_targets)):
        offsets = target - loc
        for j in range(len(loc)):
            if j == 0:
                ax.text(loc[:1, 0], loc[:1, 1], str(i))
            ax.arrow(loc[j, 0], loc[j, 1], offsets[j, 0], offsets[j, 1])

        ax.plot(loc[0:, 0], loc[0:, 1], color="g", marker="1")
        ax.plot(target[0:, 0], target[0:, 1], marker="1", color=colors[i % 5].tolist())

    plt.show()
    fig.savefig("tmp.jpg", bbox_inches="tight", pad_inches=0)
