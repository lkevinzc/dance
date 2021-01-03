import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.layers import DFConv2d, SmoothL1Loss, extreme_utils
from core.modeling.fcose.utils import get_aux_extreme_points, get_extreme_points
from core.structures import PolygonPoints
from detectron2.layers import Conv2d, DeformConv, ModulatedDeformConv, cat
from shapely.geometry import Polygon


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


class _SnakeNet(nn.Module):
    def __init__(self, cfg, stage_num):
        super(_SnakeNet, self).__init__()

        state_dim = cfg.MODEL.DANCE.HEAD.FEAT_DIM
        feature_dim = cfg.MODEL.DANCE.EDGE.CONVS_DIM + 2

        self.head = SnakeBlock(feature_dim, state_dim)

        self.res_layer_num = cfg.MODEL.DANCE.HEAD.NUM_LAYER[stage_num] - 1
        dilation = cfg.MODEL.DANCE.HEAD.CIR_DILATIONS[stage_num]
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

        back_out = self.fusion(state)
        global_state = torch.max(back_out, dim=2, keepdim=True)[0]

        global_state = global_state.expand(
            global_state.size(0), global_state.size(1), state.size(2)
        )
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x


class RefineNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        conv_dims = cfg.MODEL.DANCE.EDGE.CONVS_DIM
        prev_conv_dims = cfg.MODEL.DANCE.EDGE.CONVS_DIM
        norm = cfg.MODEL.DANCE.EDGE.NORM

        # Snake settings
        self.num_iter = cfg.MODEL.DANCE.HEAD.NUM_ITER
        self.num_convs = cfg.MODEL.DANCE.HEAD.NUM_CONVS
        self.num_sampling = cfg.MODEL.DANCE.HEAD.NUM_SAMPLING

        self.dilations = cfg.MODEL.DANCE.HEAD.DILATIONS

        self.visualize_path = cfg.MODEL.DANCE.VIS_PATH

        self.loss_reg = SmoothL1Loss(beta=cfg.MODEL.DANCE.HEAD.LOSS_L1_BETA)

        self.refine_loss_weight = cfg.MODEL.DANCE.HEAD.LOSS_WEIGHT

        # feature prep.

        self.bottom_out = nn.ModuleList()
        for i in range(self.num_convs):
            norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None

            if i == 0:
                dim_in = prev_conv_dims
            else:
                dim_in = conv_dims

            conv = Conv2d(
                dim_in,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=self.dilations[i],
                bias=not norm,
                norm=norm_module,
                activation=F.relu,
            )
            self.bottom_out.append(conv)

        # snakes
        for i in range(len(self.num_iter)):
            snake_deformer = _SnakeNet(cfg, i)
            self.__setattr__("deformer" + str(i), snake_deformer)

        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, DeformConv)
                or isinstance(m, ModulatedDeformConv)
            ):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self._logger = logging.getLogger("detectron2.utils.warnings")

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
            # for i in range(pnum):
            #     if edgenum[i] == 0:
            #         edgenum[i] = 1
            edgenum[edgenum == 0] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:
                if edgenumsum > newpnum:
                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        # take the longest and divide.
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum  # terminate
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
    def uniform_sample_1d(pts, new_n):
        if new_n == 1:
            return pts[:1]
        n = pts.shape[0]
        if n == new_n + 1:
            return pts[:-1]
        # len: n - 1
        segment_len = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))

        # down-sample or up-sample
        # n
        start_node = np.cumsum(np.concatenate([np.array([0]), segment_len]))
        total_len = np.sum(segment_len)

        new_per_len = total_len / new_n

        mark_1d = ((np.arange(new_n - 1) + 1) * new_per_len).reshape(-1, 1)
        locate = start_node.reshape(1, -1) - mark_1d
        iss, jss = np.where(locate > 0)
        cut_idx = np.cumsum(np.unique(iss, return_counts=True)[1])
        cut_idx = np.concatenate([np.array([0]), cut_idx[:-1]])

        after_idx = jss[cut_idx]
        before_idx = after_idx - 1

        after_idx[after_idx < 0] = 0

        before = locate[np.arange(new_n - 1), before_idx]
        after = locate[np.arange(new_n - 1), after_idx]

        w = (-before / (after - before)).reshape(-1, 1)

        sampled_pts = (1 - w) * pts[before_idx] + w * pts[after_idx]

        return np.concatenate([pts[:1], sampled_pts], axis=0)

    @staticmethod
    def uniform_upsample(poly, p_num):
        if poly.size(1) == 0:
            return torch.zeros([0, p_num, 2], device=poly.device), None

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

        return poly[0], edge_start_idx[0]

    def get_octagon(self, ex, p_num):
        if len(ex) == 0:
            return torch.zeros([0, p_num, 2], device=ex.device), None
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
        octagon, edge_start_idx = self.uniform_upsample(octagon, p_num)
        return octagon, edge_start_idx

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
            if instance_per_im.has("ext_points"):
                # TODO: to calibrate off-predicted boxes
                h, w = instance_per_im.image_size
                instance_per_im.ext_points.fit_to_box()
                ext_points = instance_per_im.ext_points.tensor  # (n, 4, 2)
                ext_points[..., 0].clamp_(min=0, max=w - 1)
                ext_points[..., 1].clamp_(min=0, max=h - 1)
                octagon, _ = self.get_octagon(ext_points, self.num_sampling)
                poly_sample_locations.append(octagon)
                image_index.append(ext_points.new_empty(len(ext_points)).fill_(im_i))
        if not poly_sample_locations:
            return poly_sample_locations, image_index
        poly_sample_locations = cat(poly_sample_locations, dim=0)
        image_index = cat(image_index)
        return poly_sample_locations, image_index

    def sample_bboxes_fast(self, pred_instances):
        poly_sample_locations = []
        image_index = []
        for im_i in range(len(pred_instances)):
            instance_per_im = pred_instances[im_i]
            xmin, ymin = (
                instance_per_im.pred_boxes.tensor[:, 0],
                instance_per_im.pred_boxes.tensor[:, 1],
            )  # (n,)
            xmax, ymax = (
                instance_per_im.pred_boxes.tensor[:, 2],
                instance_per_im.pred_boxes.tensor[:, 3],
            )  # (n,)
            box = [xmax, ymin, xmin, ymin, xmin, ymax, xmax, ymax]
            box = torch.stack(box, dim=1).view(-1, 4, 2)
            sampled_box, _ = self.uniform_upsample(box[None], self.num_sampling)
            poly_sample_locations.append(sampled_box)
            image_index.append(box.new_empty(len(box)).fill_(im_i))

        poly_sample_locations = cat(poly_sample_locations, dim=0)
        image_index = cat(image_index)
        return poly_sample_locations, image_index

    def sample_quad_fast(self, pred_instances):
        poly_sample_locations = []
        image_index = []
        for im_i in range(len(pred_instances)):
            instance_per_im = pred_instances[im_i]
            xmin, ymin = (
                instance_per_im.pred_boxes.tensor[:, 0],
                instance_per_im.pred_boxes.tensor[:, 1],
            )  # (n,)
            xmax, ymax = (
                instance_per_im.pred_boxes.tensor[:, 2],
                instance_per_im.pred_boxes.tensor[:, 3],
            )  # (n,)
            box = [xmax, ymin, xmin, ymin, xmin, ymax, xmax, ymax]
            box = torch.stack(box, dim=1).view(-1, 4, 2)
            sampled_box, _ = self.uniform_upsample(box[None], self.num_sampling)
            poly_sample_locations.append(sampled_box)
            image_index.append(box.new_empty(len(box)).fill_(im_i))

        quad_sample_locations = cat(poly_sample_locations, dim=0)
        image_index = cat(image_index)
        return quad_sample_locations, image_index

    def single_sample_bboxes_fast(self, pred_instances):
        instance_per_im = pred_instances[0]
        xmin, ymin = (
            instance_per_im.pred_boxes.tensor[:, 0],
            instance_per_im.pred_boxes.tensor[:, 1],
        )  # (n,)
        xmax, ymax = (
            instance_per_im.pred_boxes.tensor[:, 2],
            instance_per_im.pred_boxes.tensor[:, 3],
        )  # (n,)
        box = [xmax, ymin, xmin, ymin, xmin, ymax, xmax, ymax]
        box = torch.stack(box, dim=1).view(-1, 4, 2)
        sampled_box, _ = self.uniform_upsample(box[None], self.num_sampling)
        return sampled_box, None

    def single_sample_quad_fast(self, pred_instances):
        instance_per_im = pred_instances[0]
        xmin, ymin = (
            instance_per_im.pred_boxes.tensor[:, 0],
            instance_per_im.pred_boxes.tensor[:, 1],
        )  # (n,)
        xmax, ymax = (
            instance_per_im.pred_boxes.tensor[:, 2],
            instance_per_im.pred_boxes.tensor[:, 3],
        )  # (n,)
        box = [
            (xmax + xmin) / 2,
            ymin,
            xmin,
            (ymin + ymax) / 2,
            (xmin + xmax) / 2,
            ymax,
            xmax,
            (ymin + ymax) / 2,
        ]
        box = torch.stack(box, dim=1).view(-1, 4, 2)
        sampled_quad, _ = self.uniform_upsample(box[None], 40)
        return sampled_quad, None

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

    def compute_targets_for_polys(self, targets, image_sizes):
        poly_sample_locations = []
        poly_sample_targets = []
        dense_sample_targets = []

        init_box_locs = []
        init_ex_targets = []

        image_index = []
        scales = []

        whs = []

        up_rate = 5
        # per image
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            img_size_per_im = image_sizes[im_i]
            bboxes = targets_per_im.gt_boxes.tensor

            # no gt
            if bboxes.numel() == 0:
                continue

            gt_masks = targets_per_im.gt_masks

            # use this as a scaling
            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]

            if self.initial == "box":
                xmin, ymin = bboxes[:, 0], bboxes[:, 1]  # (n,)
                xmax, ymax = bboxes[:, 2], bboxes[:, 3]  # (n,)
                box = [xmax, ymin, xmin, ymin, xmin, ymax, xmax, ymax]
                box = torch.stack(box, dim=1).view(-1, 4, 2)
                boxes, edge_start_idx = self.uniform_upsample(
                    box[None], self.num_sampling
                )

                # just to suppress errors (DUMMY):
                init_box, _ = self.uniform_upsample(box[None], 40)
                ex_pts = init_box

            # List[np.array], element shape: (P, 2) OR None
            contours = self.get_simple_contour(gt_masks)

            # per instance
            for (oct, cnt, in_box, ex_tar, w, h, s_idx) in zip(
                boxes, contours, init_box, ex_pts, ws, hs, edge_start_idx
            ):
                if cnt is None:
                    continue

                # used for normalization
                scale = torch.min(w, h)

                # make it clock-wise
                cnt = cnt[::-1] if Polygon(cnt).exterior.is_ccw else cnt

                if Polygon(cnt).exterior.is_ccw:
                    continue

                assert not Polygon(
                    cnt
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                oct_sampled_pts = oct.cpu().numpy()
                assert not Polygon(
                    oct_sampled_pts
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                to_check = in_box.cpu().numpy()
                assert not Polygon(
                    to_check
                ).exterior.is_ccw, "0) init box must be clock-wise!"

                # sampling from ground truth
                oct_sampled_targets = self.uniform_sample(
                    cnt, len(cnt) * self.num_sampling * up_rate
                )  # (big, 2)
                # i) find a single nearest, so that becomes ordered point sets
                tt_idx = np.argmin(
                    np.power(oct_sampled_targets - oct_sampled_pts[0], 2).sum(axis=1)
                )
                oct_sampled_targets = np.roll(oct_sampled_targets, -tt_idx, axis=0)[
                    :: len(cnt)
                ]

                oct_sampled_targets, aux_ext_idxs = get_aux_extreme_points(
                    oct_sampled_targets
                )
                tt_idx = np.argmin(
                    np.power(oct_sampled_pts - oct_sampled_targets[0], 2).sum(axis=1)
                )
                oct_sampled_pts = np.roll(oct_sampled_pts, -tt_idx, axis=0)
                oct = torch.from_numpy(oct_sampled_pts).to(oct.device)
                oct_sampled_targets = self.single_uniform_multisegment_matching(
                    oct_sampled_targets, oct_sampled_pts, aux_ext_idxs, up_rate
                )
                oct_sampled_targets = torch.tensor(
                    oct_sampled_targets, device=bboxes.device
                ).float()

                oct_sampled_targets[..., 0].clamp_(min=0, max=img_size_per_im[1] - 1)
                oct_sampled_targets[..., 1].clamp_(min=0, max=img_size_per_im[0] - 1)

                dense_targets = oct_sampled_targets

                poly_sample_locations.append(oct)
                dense_sample_targets.append(dense_targets)
                poly_sample_targets.append(oct_sampled_targets)
                image_index.append(im_i)
                scales.append(scale)
                whs.append([w, h])
                init_box_locs.append(in_box)
                init_ex_targets.append(ex_tar)

        if len(init_ex_targets) > 0:
            poly_sample_locations = torch.stack(poly_sample_locations, dim=0)
            dense_sample_targets = torch.stack(dense_sample_targets, dim=0)
            poly_sample_targets = torch.stack(poly_sample_targets, dim=0)
            image_index = torch.tensor(image_index, device=bboxes.device)
            whs = torch.tensor(whs, device=bboxes.device)
            scales = torch.stack(scales, dim=0)
        else:
            return None

        return {
            "sample_locs": poly_sample_locations,
            "sample_targets": poly_sample_targets,
            "sample_dense_targets": dense_sample_targets,
            "scales": scales,
            "whs": whs,
            "image_idx": image_index,
        }

    def single_uniform_segment_matching(
        self, dense_targets, sampled_pts, edge_idx, up_rate
    ):
        """
        Several points to note while debugging:
        1) For GT (from which points are sampled), include both end by [s, e + 1] indexing.
        2) If GT not increasing (equal), shift forwards by 1.
        3) Check the 1st sampled point is indexed by 0.
        4) Check the last sampled point is NOT indexed by 0 or any small value.
        """
        ext_idx = edge_idx[::3]  # try ext first, if work then consider finer segments
        ch_pts = sampled_pts[ext_idx]  # characteristic points
        if self.initial == "box":
            # print('b', ext_idx, ch_pts)
            # tt_idx = np.argmin(np.power(sampled_pts - ch_pts[0], 2).sum(axis=1))
            # sampled_pts = np.roll(sampled_pts, -tt_idx, axis=0)
            diffs = ((ch_pts[:, None, :] - sampled_pts[None]) ** 2).sum(axis=2)
            ext_idx = np.argmin(diffs, axis=1)
            if ext_idx[0] != 0:
                ext_idx[0] = 0
            if ext_idx[3] < ext_idx[1]:
                ext_idx[3] = self.num_sampling - 1
            ch_pts = sampled_pts[ext_idx]
            # print(ext_idx, ch_pts)

        # print('im here ;)')

        aug_ext_idx = np.concatenate([ext_idx, np.array([self.num_sampling])], axis=0)

        diff = np.sum(
            (ch_pts[:, np.newaxis, :] - dense_targets[np.newaxis, :, :]) ** 2, axis=2
        )
        min_idx = np.argmin(diff, axis=1)

        aug_min_idx = np.concatenate(
            [min_idx, np.array([self.num_sampling * up_rate])], axis=0
        )

        if aug_min_idx[3] < aug_min_idx[1]:
            # self._logger.info("WARNING: Last point not matching!")
            # self._logger.info(aug_ext_idx)
            # self._logger.info(aug_min_idx)
            aug_min_idx[3] = (
                self.num_sampling * up_rate - 2
            )  # enforce matching of the last point

        if aug_min_idx[0] != 0:
            # TODO: This is crucial, or other wise the first point may be
            # TODO: matched to near 640, then sorting will completely mess
            # self._logger.info("WARNING: First point not matching!")
            # self._logger.info(aug_ext_idx)
            # self._logger.info(aug_min_idx)
            aug_min_idx[0] = 0  # enforce matching of the 1st point

        aug_ext_idx = np.sort(aug_ext_idx)
        aug_min_idx = np.sort(aug_min_idx)

        # === error-prone ===

        # deal with corner cases

        if aug_min_idx[2] == self.num_sampling * up_rate - 1:
            # print("WARNING: Bottom extreme point being the last point!")
            self._logger.info("WARNING: Bottom extreme point being the last point!")
            # hand designed remedy
            aug_min_idx[2] = self.num_sampling * up_rate - 3
            aug_min_idx[3] = self.num_sampling * up_rate - 2

        if aug_min_idx[3] == self.num_sampling * up_rate - 1:
            # print("WARNING: Right extreme point being the last point!")
            # self._logger.info("WARNING: Right extreme point being the last point!")
            # self._logger.info(aug_ext_idx)
            # self._logger.info(aug_min_idx)
            aug_min_idx[3] -= 1
            aug_min_idx[2] -= 1

        segments = []
        try:
            for i in range(4):
                if aug_ext_idx[i + 1] - aug_ext_idx[i] == 0:
                    continue  # no need to sample for this segment

                if aug_min_idx[i + 1] - aug_min_idx[i] <= 0:
                    # overlap due to quantization, negative value is due to accumulation of overlap
                    aug_min_idx[i + 1] = aug_min_idx[i] + 1  # guarantee spacing

                if i == 3:  # last, complete a circle
                    pts = np.concatenate(
                        [dense_targets[aug_min_idx[i] :], dense_targets[:1]], axis=0
                    )
                else:
                    pts = dense_targets[
                        aug_min_idx[i] : aug_min_idx[i + 1] + 1
                    ]  # including
                new_sampled_pts = self.uniform_sample_1d(
                    pts, aug_ext_idx[i + 1] - aug_ext_idx[i]
                )
                segments.append(new_sampled_pts)
            # segments.append(dense_targets[-1:]) # close the loop
            segments = np.concatenate(segments, axis=0)
            if len(segments) != self.num_sampling:
                # print("WARNING: Number of points not matching!")
                self._logger.info(
                    "WARNING: Number of points not matching!", len(segments)
                )
                raise ValueError(len(segments))
        except Exception as err:  # may exist some very tricky corner cases...
            # print("WARNING: Tricky corner cases occurred!")
            self._logger.info("WARNING: Tricky corner cases occurred!")
            self._logger.info(err)
            self._logger.info(aug_ext_idx)
            self._logger.info(aug_min_idx)
            # raise ValueError('TAT')
            segments = self.reorder_perloss(
                torch.from_numpy(dense_targets[::up_rate][None]),
                torch.from_numpy(sampled_pts)[None],
            )[0]
            segments = segments.numpy()

        return segments

    def single_uniform_multisegment_matching(
        self, dense_targets, sampled_pts, ext_idx, up_rate
    ):
        """
        Several points to note while debugging:
        1) For GT (from which points are sampled), include both end by [s, e + 1] indexing.
        2) If GT not increasing (equal), shift forwards by 1.
        3) Check the 1st sampled point is indexed by 0.
        4) Check the last sampled point is NOT indexed by 0 or any small value.
        """
        min_idx = ext_idx

        ch_pts = dense_targets[min_idx]  # characteristic points

        diffs = ((ch_pts[:, np.newaxis, :] - sampled_pts[np.newaxis]) ** 2).sum(axis=2)
        ext_idx = np.argmin(diffs, axis=1)
        if ext_idx[0] != 0:
            ext_idx[0] = 0
        if ext_idx[-1] < ext_idx[1]:
            ext_idx[-1] = self.num_sampling - 2
        ext_idx = np.sort(ext_idx)

        aug_ext_idx = np.concatenate([ext_idx, np.array([self.num_sampling])], axis=0)

        # diff = np.sum((ch_pts[:, np.newaxis, :] - dense_targets[np.newaxis, :, :]) ** 2, axis=2)
        # min_idx = np.argmin(diff, axis=1)

        aug_min_idx = np.concatenate(
            [min_idx, np.array([self.num_sampling * up_rate])], axis=0
        )

        if aug_min_idx[-1] < aug_min_idx[1]:
            self._logger.info("WARNING: Last point not matching!")
            self._logger.info(aug_ext_idx)
            self._logger.info(aug_min_idx)
            aug_min_idx[-1] = (
                self.num_sampling * up_rate - 2
            )  # enforce matching of the last point

        if aug_min_idx[0] != 0:
            # TODO: This is crucial, or other wise the first point may be
            # TODO: matched to near 640, then sorting will completely mess
            self._logger.info("WARNING: First point not matching!")
            self._logger.info(aug_ext_idx)
            self._logger.info(aug_min_idx)
            aug_min_idx[0] = 0  # enforce matching of the 1st point

        aug_ext_idx = np.sort(aug_ext_idx)
        aug_min_idx = np.sort(aug_min_idx)

        # === error-prone ===

        # deal with corner cases

        if aug_min_idx[-2] == self.num_sampling * up_rate - 1:
            # print("WARNING: Bottom extreme point being the last point!")
            self._logger.info("WARNING: Bottom extreme point being the last point!")
            # hand designed remedy
            aug_min_idx[-2] = self.num_sampling * up_rate - 3
            aug_min_idx[-1] = self.num_sampling * up_rate - 2

        if aug_min_idx[-1] == self.num_sampling * up_rate - 1:
            # print("WARNING: Right extreme point being the last point!")
            self._logger.info("WARNING: Right extreme point being the last point!")
            self._logger.info(aug_ext_idx)
            self._logger.info(aug_min_idx)
            aug_min_idx[-1] -= 1
            aug_min_idx[-2] -= 1

        segments = []
        try:
            for i in range(len(ext_idx)):
                if aug_ext_idx[i + 1] - aug_ext_idx[i] == 0:
                    continue  # no need to sample for this segment

                if aug_min_idx[i + 1] - aug_min_idx[i] <= 0:
                    # overlap due to quantization, negative value is due to accumulation of overlap
                    aug_min_idx[i + 1] = aug_min_idx[i] + 1  # guarantee spacing

                if i == len(ext_idx) - 1:  # last, complete a circle
                    pts = np.concatenate(
                        [dense_targets[aug_min_idx[i] :], dense_targets[:1]], axis=0
                    )
                else:
                    pts = dense_targets[
                        aug_min_idx[i] : aug_min_idx[i + 1] + 1
                    ]  # including
                new_sampled_pts = self.uniform_sample_1d(
                    pts, aug_ext_idx[i + 1] - aug_ext_idx[i]
                )
                segments.append(new_sampled_pts)
            # segments.append(dense_targets[-1:]) # close the loop
            segments = np.concatenate(segments, axis=0)
            if len(segments) != self.num_sampling:
                # print("WARNING: Number of points not matching!")
                self._logger.info(
                    "WARNING: Number of points not matching!", len(segments)
                )
                raise ValueError(len(segments))
        except Exception as err:  # may exist some very tricky corner cases...
            # print("WARNING: Tricky corner cases occurred!")
            self._logger.info("WARNING: Tricky corner cases occurred!")
            self._logger.info(err)
            self._logger.info(aug_ext_idx)
            self._logger.info(aug_min_idx)
            # raise ValueError('TAT')
            segments = self.reorder_perloss(
                torch.from_numpy(dense_targets[::up_rate][None]),
                torch.from_numpy(sampled_pts)[None],
            )[0]
            segments = segments.numpy()

        return segments

    def reorder_perloss(self, oct_sampled_targets, oct_sampled_pts):
        """
        Adaptively adjust the penalty, concept-wise the loss is much more reasonable.
        :param oct_sampled_targets: (\sum{k}, num_sampling, 2) for all instances
        :param oct_sampled_pts: same~
        :return:
        """
        assert oct_sampled_targets.size() == oct_sampled_pts.size()
        n = len(oct_sampled_targets)
        num_locs = oct_sampled_pts.size(1)
        ind1 = torch.arange(num_locs, device=oct_sampled_targets.device)
        ind2 = ind1.expand(num_locs, -1)
        enumerated_ind = torch.fmod(ind2 + ind1.view(-1, 1), num_locs).view(-1).long()
        enumerated_targets = oct_sampled_targets[:, enumerated_ind, :].view(
            n, -1, num_locs, 2
        )
        diffs = enumerated_targets - oct_sampled_pts[:, None, ...]
        diffs_sum = diffs.pow(2).sum(3).sum(2)
        tt_idx = torch.argmin(diffs_sum, dim=1)
        re_ordered_gt = enumerated_targets[torch.arange(n), tt_idx]
        return re_ordered_gt

    def get_locations_feature(self, features, locations, image_idx):
        h = features.shape[2] * 4
        w = features.shape[3] * 4
        locations = locations.clone()
        locations[..., 0] = locations[..., 0] / (w / 2.0) - 1
        locations[..., 1] = locations[..., 1] / (h / 2.0) - 1

        batch_size = features.size(0)
        sampled_features = torch.zeros(
            [locations.size(0), features.size(1), locations.size(1)],
            device=locations.device,
        )
        for i in range(batch_size):
            if image_idx is None:
                per_im_loc = locations.unsqueeze(0)
            else:
                per_im_loc = locations[image_idx == i].unsqueeze(0)
            feature = torch.nn.functional.grid_sample(
                features[i : i + 1],
                per_im_loc,
                padding_mode="reflection",
                align_corners=False,
            )[0].permute(1, 0, 2)
            if image_idx is None:
                sampled_features = feature
            else:
                sampled_features[image_idx == i] = feature

        return sampled_features

    @staticmethod
    def de_location(locations):
        # de-location (spatial relationship among locations; translation invariant)
        x_min = torch.min(locations[..., 0], dim=-1)[0]
        y_min = torch.min(locations[..., 1], dim=-1)[0]
        x_max = torch.max(locations[..., 0], dim=-1)[0]
        y_max = torch.max(locations[..., 1], dim=-1)[0]
        new_locations = locations.clone()
        new_locations[..., 0] = (new_locations[..., 0] - x_min[..., None]) / (
            x_max[..., None] - x_min[..., None]
        )
        new_locations[..., 1] = (new_locations[..., 1] - y_min[..., None]) / (
            y_max[..., None] - y_min[..., None]
        )
        return new_locations

    def evolve(
        self,
        deformer,
        features,
        locations,
        image_idx,
        image_sizes,
        whs,
        att=False,
    ):

        locations_for_sample = locations

        sampled_features = self.get_locations_feature(
            features, locations_for_sample, image_idx
        )

        att_scores = sampled_features[:, :1, :]
        sampled_features = sampled_features[:, 1:, :]

        calibrated_locations = self.de_location(locations_for_sample)

        # (Sigma{num_poly_i}, 2 + feat_dim, 128)
        concat_features = torch.cat(
            [sampled_features, calibrated_locations.permute(0, 2, 1)], dim=1
        )

        pred_offsets = deformer(concat_features)

        pred_offsets = pred_offsets.permute(0, 2, 1)

        pred_offsets = torch.tanh(pred_offsets) * whs[:, None, :]

        pred_offsets = pred_offsets * att_scores.permute(0, 2, 1)

        pred_locations = locations + pred_offsets

        self.clip_locations(pred_locations, image_idx, image_sizes)

        return pred_locations

    @staticmethod
    def clip_locations(pred_locs, image_idx, image_sizes):
        if image_idx is None:
            pred_locs[0, :, 0::2].clamp_(min=0, max=image_sizes[1] - 1)
            pred_locs[0, :, 1::2].clamp_(min=0, max=image_sizes[0] - 1)
        else:
            for i, img_size_per_im in enumerate(image_sizes):
                pred_locs[image_idx == i, :, 0::2].clamp_(
                    min=0, max=img_size_per_im[1] - 1
                )
                pred_locs[image_idx == i, :, 1::2].clamp_(
                    min=0, max=img_size_per_im[0] - 1
                )

    def single_test(self, features, pred_instances):

        locations, image_idx = self.single_sample_bboxes_fast(pred_instances)

        if len(locations) == 0:
            return pred_instances, {}

        image_sizes = pred_instances[0].image_size
        # print(image_sizes)
        # print(features.shape)
        # bboxes = pred_instances[0].pred_boxes.tensor
        # print(bboxes.max(0)[0], bboxes.min(0)[0])
        pred_instances[0].pred_boxes.clip(image_sizes)
        bboxes = pred_instances[0].pred_boxes.tensor
        ws = bboxes[:, 2] - bboxes[:, 0]
        hs = bboxes[:, 3] - bboxes[:, 1]
        whs = torch.stack([ws, hs], dim=1)

        edge_band = features[:, :1, ...]
        features = features[:, 1:, ...]

        for i in range(self.num_convs):
            features = self.bottom_out[i](features)

        features = torch.cat([edge_band, features], dim=1)
        location_preds = []

        for i in range(len(self.num_iter)):
            deformer = self.__getattr__("deformer" + str(i))
            if i == 0:
                pred_location = self.evolve(
                    deformer, features, locations, image_idx, image_sizes, whs
                )
            else:
                pred_location = self.evolve(
                    deformer,
                    features,
                    pred_location,
                    image_idx,
                    image_sizes,
                    whs,
                    att=True,
                )
            location_preds.append(pred_location)

        pred_per_im = location_preds[-1]
        instance_per_im = pred_instances[0]
        instance_per_im.pred_polys = PolygonPoints(pred_per_im)

        return [instance_per_im], {}

    def forward(self, features, pred_instances=None, targets=None):
        # start = time.time()
        if self.training:
            gt_instances = targets[0]
            image_sizes = targets[1]
            training_targets = self.compute_targets_for_polys(gt_instances, image_sizes)
            if training_targets is None:
                return [], {
                    "loss_stage_0": 0,
                    "loss_stage_1": 0,
                    "loss_stage_2": 0,
                    "loss_edge_det": 0,
                }
            locations, reg_targets, scales, image_idx = (
                training_targets["sample_locs"],
                training_targets["sample_targets"],
                training_targets["scales"],
                training_targets["image_idx"],
            )
            whs = training_targets["whs"]
        else:
            if not self.visualize_path:
                return self.single_test(features, pred_instances)

            assert pred_instances is not None

            locations, image_idx = self.sample_bboxes_fast(pred_instances)
            if len(locations) == 0:
                return pred_instances, {}
            image_sizes = list(map(lambda x: x.image_size, pred_instances))
            bboxes = list(map(lambda x: x.pred_boxes.tensor, pred_instances))
            bboxes = cat(bboxes, dim=0)

            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]
            whs = torch.stack([ws, hs], dim=1)

        edge_band = features[:, :1, ...]
        features = features[:, 1:, ...]

        # feature preparation
        for i in range(self.num_convs):
            features = self.bottom_out[i](features)

        features = torch.cat([edge_band, features], dim=1)
        location_preds = []
        for i in range(len(self.num_iter)):
            deformer = self.__getattr__("deformer" + str(i))
            if i == 0:
                pred_location, multi_pred_locations = self.evolve(
                    deformer,
                    features,
                    locations,
                    image_idx,
                    image_sizes,
                    whs,
                )
            else:
                pred_location, multi_pred_locations = self.evolve(
                    deformer,
                    features,
                    pred_location,
                    image_idx,
                    image_sizes,
                    whs,
                    att=True,
                )
            location_preds.append(pred_location)

        if self.training:
            loss = {}
            for i, pred in enumerate(zip(location_preds)):
                loss_name = "loss_stage_" + str(i)
                stage_weight = 1 / 3
                loss_func = self.loss_reg

                dynamic_reg_targets = reg_targets

                point_weight = (
                    torch.tensor(1, device=scales.device).float() / whs[:, None, :]
                )

                stage_loss = (
                    loss_func(pred * point_weight, dynamic_reg_targets * point_weight)
                    * stage_weight
                )

                loss[loss_name] = stage_loss * self.refine_loss_weight

            return [], loss
        else:
            new_instances = self.predict_postprocess(
                pred_instances,
                locations,
                location_preds,
                image_idx,
            )
            return new_instances, {}

    def predict_postprocess(
        self,
        pred_instances,
        locations,
        location_preds,
        image_idx,
        scores=None,
    ):
        if image_idx is None:
            pred_per_im = location_preds[-1]
            instance_per_im = pred_instances[0]
            instance_per_im.pred_polys = PolygonPoints(pred_per_im)
            return [instance_per_im]

        results = []
        # per im
        for i, instance_per_im in enumerate(pred_instances):
            pred_per_im = location_preds[-1][image_idx == i]  # N x 128 x 2

            if scores is not None:
                score_per_im = scores[image_idx == i]
                instance_per_im.scores *= score_per_im

            instance_per_im.pred_polys = PolygonPoints(pred_per_im)

            if self.visualize_path:
                path = []
                loc_per_im = locations[image_idx == i]
                path.append(loc_per_im)

                for k in range(len(self.num_iter)):
                    nodes_per_im = location_preds[k][image_idx == i]
                    if nodes_per_im.size(1) == self.num_sampling:
                        path.append(nodes_per_im)

                path = torch.stack(path, dim=1)
                instance_per_im.pred_path = path

            results.append(instance_per_im)
        return results


def vis(image, poly_sample_locations, poly_sample_targets):
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


def vis_single(image, pts):
    import matplotlib.pyplot as plt

    image = image.cpu().numpy().transpose(1, 2, 0)[:, :, ::-1].astype(np.uint8)
    pts = pts.cpu().numpy()[0]

    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.tight_layout()

    ax.imshow(image)

    ax.plot(pts[0:, 0], pts[0:, 1], color="g", marker="1")

    plt.show()
    fig.savefig("tmp.jpg", bbox_inches="tight", pad_inches=0)
