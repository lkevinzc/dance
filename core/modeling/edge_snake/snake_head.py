import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.layers import Conv2d, DeformConv, ModulatedDeformConv, cat

import logging
import time
from core.layers import DFConv2d, SmoothL1Loss, extreme_utils
from core.modeling.fcose.utils import get_aux_extreme_points, get_extreme_points
from core.structures import ExtremePoints, PolygonPoints
from core.utils import timer
from shapely.geometry import Polygon


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat(
            [
                input_tensor,
                xx_channel.type_as(input_tensor),
                yy_channel.type_as(input_tensor),
            ],
            dim=1,
        )

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)

        return ret


def _close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def _compute_iou(poly_t, poly_d, min_area):
    poly_t = Polygon(poly_t)
    poly_d = Polygon(poly_d)
    i = poly_t.intersection(poly_d).area
    u = poly_t.union(poly_d).area
    valid = 1 if poly_t.area > min_area else 0
    return [valid, i / u]


def _polygons_to_mask(polygons, h, w):
    rle = mask_util.frPyObjects(polygons, h, w)
    rle = mask_util.merge(rle)
    return mask_util.decode(rle)[:, :]


def _compute_iou_coco(poly_t, poly_d, min_area):
    length = max(poly_t.max(), poly_d.max()) + 10
    m_t = _polygons_to_mask([poly_t.tolist()], length, length)
    m_d = _polygons_to_mask([poly_d.tolist()], length, length)
    i = np.sum(m_t * m_d)
    u = np.sum(m_t) + np.sum(m_d) - i
    valid = 1 if np.sum(m_t) > min_area else 0
    return [valid, i / u]


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


class _MSnakeNet(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(_MSnakeNet, self).__init__()

        state_dim = cfg.MODEL.SNAKE_HEAD.MSCORE_SNAKE_FEAT_DIM
        feature_dim = cfg.MODEL.EDGE_HEAD.CONVS_DIM + 2

        if cfg.MODEL.SNAKE_HEAD.MARK_INDEX:
            feature_dim += 1

        self.head = SnakeBlock(feature_dim, state_dim)

        self.res_layer_num = cfg.MODEL.SNAKE_HEAD.MSCORE_SNAKE_NUM_LAYER - 1
        dilation = cfg.MODEL.SNAKE_HEAD.MSCORE_SNAKE_CIR_DILATIONS

        for i in range(self.res_layer_num):
            conv = SnakeBlock(state_dim, state_dim, n_adj=4, dilation=dilation[i])
            self.__setattr__("res" + str(i), conv)

        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim, state_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(state_dim, 80, 1),
        )

    def forward(self, x):
        states = []

        x = self.head(x)
        # states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__("res" + str(i))(x) + x
            # states.append(x)

        global_state = torch.mean(x, dim=2, keepdim=True)
        x = self.prediction(global_state)

        # state = torch.cat(states, dim=1)
        # global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        # global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        # state = torch.cat([global_state, state], dim=1)
        # x = self.prediction(state)

        return x


class _SnakeNet(nn.Module):
    def __init__(self, cfg, stage_num):
        super(_SnakeNet, self).__init__()

        state_dim = cfg.MODEL.SNAKE_HEAD.FEAT_DIM
        feature_dim = cfg.MODEL.EDGE_HEAD.CONVS_DIM + 2

        if cfg.MODEL.SNAKE_HEAD.MARK_INDEX:
            feature_dim += 1

        if cfg.MODEL.SNAKE_HEAD.TRACK_PATH and stage_num > 0:
            feature_dim += 2

        self.head = SnakeBlock(feature_dim, state_dim)

        self.num_offset = cfg.MODEL.SNAKE_HEAD.MULTI_OFFSET

        self.res_layer_num = cfg.MODEL.SNAKE_HEAD.NUM_LAYER[stage_num] - 1
        dilation = cfg.MODEL.SNAKE_HEAD.CIR_DILATIONS[stage_num]
        for i in range(self.res_layer_num):
            conv = SnakeBlock(state_dim, state_dim, n_adj=4, dilation=dilation[i])
            self.__setattr__("res" + str(i), conv)

        fusion_state_dim = 256
        self.skip = cfg.MODEL.SNAKE_HEAD.SKIP

        # if self.skip:
        #     fusion_state_dim = feature_dim

        self.fusion = nn.Conv1d(
            state_dim * (self.res_layer_num + 1), fusion_state_dim, 1
        )
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2 * self.num_offset, 1),
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

        # # big skip to conn spatial feat from 2D conv.
        # if self.skip:
        #     back_out += x

        global_state = global_state.expand(
            global_state.size(0), global_state.size(1), state.size(2)
        )
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)

        return x


class _PreOffSnakeNet(nn.Module):
    def __init__(self, cfg):
        super(_PreOffSnakeNet, self).__init__()

        state_dim = cfg.MODEL.SNAKE_HEAD.FEAT_DIM
        feature_dim = cfg.MODEL.EDGE_HEAD.CONVS_DIM + 2

        if cfg.MODEL.SNAKE_HEAD.MARK_INDEX:
            feature_dim += 1

        self.head = SnakeBlock(feature_dim, state_dim)

        self.res_layer_num = cfg.MODEL.SNAKE_HEAD.NUM_LAYER[0] - 1
        dilation = cfg.MODEL.SNAKE_HEAD.CIR_DILATIONS[0]
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
        self.global_prediction = nn.Sequential(
            nn.Conv1d(fusion_state_dim, state_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(state_dim, 2, 1),
        )

    def forward(self, x):
        states = []

        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__("res" + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)
        fused_fea = self.fusion(state)
        global_state = torch.max(fused_fea, dim=2, keepdim=True)[0]

        mean_state = torch.mean(fused_fea, dim=2, keepdim=True)

        global_state = global_state.expand(
            global_state.size(0), global_state.size(1), state.size(2)
        )
        state = torch.cat([global_state, state], dim=1)
        x = self.prediction(state)
        offset = self.global_prediction(mean_state)
        return x, offset


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


def _make_aspp(in_features, out_features, dilation):
    if dilation == 1:
        conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False)
    else:
        conv = nn.Conv2d(
            in_features,
            out_features,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
    bn = nn.BatchNorm2d(out_features, affine=True)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)


def _make_psp(in_features, size):
    prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
    conv = nn.Conv2d(in_features, in_features // 4, kernel_size=1, bias=False)
    bn = nn.BatchNorm2d(in_features // 4, affine=True)
    relu = nn.ReLU(inplace=True)

    return nn.Sequential(prior, conv, bn, relu)


def _make_image_fea(in_features, out_features):
    g_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    conv = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False)
    bn = nn.BatchNorm2d(out_features, affine=True)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(g_pool, conv, bn, relu)


def _make_block(in_features, out_features, ks=1, pad=0):
    conv = nn.Conv2d(
        in_features, out_features, kernel_size=ks, padding=pad, stride=1, bias=False
    )
    bn = nn.BatchNorm2d(out_features, affine=True)
    relu = nn.ReLU(inplace=True)
    return nn.Sequential(conv, bn, relu)


class RefineNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.add_coords = AddCoords()
        self.coord_conv = cfg.MODEL.SNAKE_HEAD.COORD_CONV

        self.detach = cfg.MODEL.SNAKE_HEAD.DETACH

        conv_dims = cfg.MODEL.EDGE_HEAD.CONVS_DIM
        prev_conv_dims = cfg.MODEL.EDGE_HEAD.CONVS_DIM
        norm = cfg.MODEL.EDGE_HEAD.NORM
        df_conv_dims = cfg.MODEL.DEFORM_HEAD.NUM_CONVS

        # Mask scoring snake
        if cfg.MODEL.SNAKE_HEAD.MSCORE_SNAKE_ON:
            norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
            self.ms_layer = Conv2d(
                prev_conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not norm,
                norm=norm_module,
                activation=F.relu,
            )
            self.ms_head = _MSnakeNet(cfg)
            self.ms_min_area = cfg.MODEL.SNAKE_HEAD.MSCORE_SNAKE_MIN_AREA
            self.ms_loss_weight = cfg.MODEL.SNAKE_HEAD.MSCORE_SNAKE_LOSS_WEIGHT
        else:
            self.ms_head = None

        # Snake settings
        self.structure = cfg.MODEL.SNAKE_HEAD.STRUCTURE
        self.conv_type = cfg.MODEL.SNAKE_HEAD.CONV_TYPE
        self.initial = cfg.MODEL.SNAKE_HEAD.INITIAL
        self.num_adj = cfg.MODEL.SNAKE_HEAD.FILTER_WIDTH
        self.num_iter = cfg.MODEL.SNAKE_HEAD.NUM_ITER
        self.num_convs = cfg.MODEL.SNAKE_HEAD.NUM_CONVS
        self.num_sampling = cfg.MODEL.SNAKE_HEAD.NUM_SAMPLING
        self.mark_index = cfg.MODEL.SNAKE_HEAD.MARK_INDEX

        self.de_location_type = cfg.MODEL.SNAKE_HEAD.DE_LOC_TYPE
        self.individual_scale = cfg.MODEL.SNAKE_HEAD.INDIVIDUAL_SCALE
        self.dilations = cfg.MODEL.SNAKE_HEAD.DILATIONS
        self.reorder_method = cfg.MODEL.SNAKE_HEAD.REORDER_METHOD

        self.jittering = cfg.MODEL.SNAKE_HEAD.JITTERING

        self.last_up_sample = cfg.MODEL.SNAKE_HEAD.LAST_UP_SAMPLE
        self.up_sample_rate = cfg.MODEL.SNAKE_HEAD.UP_SAMPLE_RATE
        self.last_chamfer = cfg.MODEL.SNAKE_HEAD.LAST_CHAMFER

        self.track_path = cfg.MODEL.SNAKE_HEAD.TRACK_PATH

        self.visualize_path = cfg.MODEL.SNAKE_HEAD.VIS_PATH

        self.original = cfg.MODEL.SNAKE_HEAD.ORIGINAL

        if cfg.MODEL.SNAKE_HEAD.LOSS_TYPE == "smoothl1":
            self.loss_reg = SmoothL1Loss(beta=cfg.MODEL.SNAKE_HEAD.LOSS_L1_BETA)
        elif cfg.MODEL.SNAKE_HEAD.LOSS_TYPE == "chamfer":
            self.loss_reg = ChamferLoss()

        if cfg.MODEL.SNAKE_HEAD.LOSS_SEPARATE_REFINE:
            self.loss_refine = SmoothL1Loss(beta=0.033)
        elif self.last_chamfer:
            self.loss_refine = ChamferLoss()
            if cfg.MODEL.SNAKE_HEAD.LAST_NEIGHBOR:
                self.loss_neighbor = SmoothL1Loss(beta=0.01)
        else:
            self.loss_refine = None

        if not cfg.MODEL.SNAKE_HEAD.LAST_NEIGHBOR:
            self.loss_neighbor = None

        if cfg.MODEL.SNAKE_HEAD.LAST_CHAMFER:
            self.loss_last_chamfer = ChamferLoss()

        self.multi_offset = cfg.MODEL.SNAKE_HEAD.MULTI_OFFSET

        self.new_matching = cfg.MODEL.SNAKE_HEAD.NEW_MATCHING

        self.loss_adaptive = cfg.MODEL.SNAKE_HEAD.LOSS_ADAPTIVE
        loss_distribution = F.softmax(
            torch.tensor(cfg.MODEL.SNAKE_HEAD.LOSS_DISTRIBUTION).float(), dim=0
        )
        self.stage_loss_weigh = cfg.MODEL.SNAKE_HEAD.LOSS_WEIGH
        if self.stage_loss_weigh:
            self.loss_distribution = cfg.MODEL.SNAKE_HEAD.LOSS_DISTRIBUTION
        self.point_loss_weight = cfg.MODEL.SNAKE_HEAD.POINT_WEIGH
        self.selective_refine = cfg.MODEL.SNAKE_HEAD.SELECTIVE_REFINE
        self.double_selective_refine = cfg.MODEL.SNAKE_HEAD.DOUBLE_SELECTIVE_REFINE

        self.pred_edge = cfg.MODEL.SNAKE_HEAD.PRED_EDGE

        refine_loss_type = cfg.MODEL.SNAKE_HEAD.LOSS_TYPE
        refine_loss_weight = (
            7 if refine_loss_type == "chamfer" else 10
        )  # 7 normal; 3 sqrt
        point_weight = cfg.MODEL.SNAKE_HEAD.POINT_WEIGH
        self.refine_loss_weight = 0.2 if point_weight else refine_loss_weight
        self.edge_in_separate = cfg.MODEL.SNAKE_HEAD.EDGE_IN_SEPARATE

        self.use_aspp = cfg.MODEL.SNAKE_HEAD.USE_ASPP
        self.use_psp = cfg.MODEL.SNAKE_HEAD.USE_PSP
        self.pre_offset = cfg.MODEL.SNAKE_HEAD.PRE_OFFSET

        self.attention = cfg.MODEL.SNAKE_HEAD.ATTENTION

        # feature prep.
        if self.use_aspp:
            # use aspp to get more diverse spatial features
            c_dim = cfg.MODEL.SNAKE_HEAD.ASPP_DIM
            dilations = cfg.MODEL.SNAKE_HEAD.ASPP_DILATIONS
            self.aspp = nn.ModuleList(
                [_make_aspp(prev_conv_dims, c_dim, dilation) for dilation in dilations]
            )
            self.image_pool = _make_image_fea(prev_conv_dims, c_dim)
            self.encode_fea = _make_block(5 * c_dim, conv_dims)
        elif self.use_psp:
            # use psp to get more diverse spatial features
            pooled_sizes = cfg.MODEL.SNAKE_HEAD.PSP_SIZE
            self.psp = nn.ModuleList(
                [_make_psp(prev_conv_dims, size) for size in pooled_sizes]
            )
            self.encode_fea = _make_block(2 * prev_conv_dims, conv_dims)
        else:
            self.bottom_out = nn.ModuleList()
            for i in range(self.num_convs):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None

                if i == 0:
                    dim_in = prev_conv_dims
                else:
                    dim_in = conv_dims

                if (
                    i == 0 and (cfg.MODEL.SNAKE_HEAD.EDGE_IN or self.pred_edge)
                ) or self.edge_in_separate[i]:
                    dim_in += 1

                if self.coord_conv[i]:
                    dim_in += 2

                if cfg.MODEL.SNAKE_HEAD.USE_DEFORMABLE[i]:
                    conv = DFConv2d(
                        dim_in,
                        df_conv_dims,
                        kernel_size=3,
                        stride=1,
                        bias=not norm,
                        norm=norm_module,
                        activation=F.relu,
                    )
                else:
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
                    if cfg.MODEL.SNAKE_HEAD.STRONGER:
                        extra_conv = Conv2d(
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
                        conv = nn.Sequential(conv, extra_conv)
                # weight_init.c2_msra_fill(conv) - DONT USE THIS INIT
                self.bottom_out.append(conv)

        if self.original:
            self.init_deformer = _SnakeNet(cfg, 0)

        # snakes
        for i in range(len(self.num_iter)):
            if i == 0 and self.pre_offset:
                snake_deformer = _PreOffSnakeNet(cfg)
            elif self.conv_type == "ccn":
                snake_deformer = _SnakeNet(cfg, i)
            elif self.conv_type == "gcn":
                snake_deformer = _ResGCNNet(state_dim=128, feature_dim=conv_dims + 2)
            else:
                raise ValueError("Unsupported operation!")
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

        # hand inserted logger, for debugging
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

    # def simpler_sample_bboxes_fast(self, pred_instances):
    #     poly_sample_locations = []
    #     image_index = []
    #     for im_i in range(len(pred_instances)):
    #         instance_per_im = pred_instances[im_i]
    #         xmin, ymin = instance_per_im.pred_boxes.tensor[:, 0], instance_per_im.pred_boxes.tensor[:, 1]  # (n,)
    #         xmax, ymax = instance_per_im.pred_boxes.tensor[:, 2], instance_per_im.pred_boxes.tensor[:, 3]  # (n,)
    #
    #         perimeter = 2 * ((xmax - xmin) + (ymax - ymin))
    #
    #
    #         box = torch.stack(box, dim=1).view(-1, 4, 2)
    #         sampled_box, _ = self.uniform_upsample(box[None], self.num_sampling)
    #         poly_sample_locations.append(sampled_box)
    #         image_index.append(box.new_empty(len(box)).fill_(im_i))

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
        dense_sample_targets = []  # 3x number of sampling

        init_box_locs = []
        init_ex_targets = []

        edge_index = []
        image_index = []
        scales = []
        # cls = []

        whs = []

        if self.new_matching:
            up_rate = 5  # TODO: subject to change, (hard-code 5x.
        else:
            up_rate = 1
        # up_rate = 5

        # per image
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            img_size_per_im = image_sizes[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            # classes = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                continue

            gt_masks = targets_per_im.gt_masks

            # use this as a scaling
            ws = bboxes[:, 2] - bboxes[:, 0]
            hs = bboxes[:, 3] - bboxes[:, 1]

            if (self.initial == "octagon") or (self.original):
                # [t_H_off, l_V_off, b_H_off, r_V_off]
                ext_pts_off = self.get_simple_extreme_points(gt_masks.polygons).to(
                    bboxes.device
                )
                ex_t = torch.stack([ext_pts_off[:, None, 0], bboxes[:, None, 1]], dim=2)
                ex_l = torch.stack([bboxes[:, None, 0], ext_pts_off[:, None, 1]], dim=2)
                ex_b = torch.stack([ext_pts_off[:, None, 2], bboxes[:, None, 3]], dim=2)
                ex_r = torch.stack([bboxes[:, None, 2], ext_pts_off[:, None, 3]], dim=2)
                ex_pts = torch.cat([ex_t, ex_l, ex_b, ex_r], dim=1)
                # print('correct')
                # if self.jittering:
                #     turbulence = torch.randn_like(ex_pts, device=bboxes.device) * self.jittering
                #
                #     # for box
                #     ex_pts[:, 1::2, 0] += turbulence[:, 1::2, 0] * ws[:, None]
                #     ex_pts[:, 0::2, 1] += turbulence[:, 0::2, 1] * hs[:, None]
                #     # for ext
                #     ex_pts[:, 0::2, 0] += turbulence[:, 0::2, 0] * ws[:, None] * 0.25
                #     ex_pts[:, 1::2, 1] += turbulence[:, 1::2, 1] * hs[:, None] * 0.25
                #
                #     ex_pts[..., 0].clamp_(min=0, max=img_size_per_im[1] - 1)
                #     ex_pts[..., 1].clamp_(min=0, max=img_size_per_im[0] - 1)

                # N x num_sampling x 2 ; N x num_sampling. (By GPU!)
                octagons, edge_start_idx = self.get_octagon(ex_pts, self.num_sampling)
                aux_octagons = octagons  # to get ext points
                # N x 16 (ccw)
                # octagons = ExtremePoints(ex_pts).get_octagons().cpu().numpy().reshape(-1, 8, 2)

                if self.original:
                    xmin, ymin = bboxes[:, 0], bboxes[:, 1]  # (n,)
                    xmax, ymax = bboxes[:, 2], bboxes[:, 3]  # (n,)
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
                    init_box, _ = self.uniform_upsample(box[None], 40)
                    # init_tar = torch.cat([ex_t, ex_l, ex_b, ex_r], dim=1)

            if (self.initial == "box") and (not self.original):
                # upper_right = torch.stack([bboxes[:, None, 2], bboxes[:, None, 1]], dim=2)
                # upper_left = torch.stack([bboxes[:, None, 0], bboxes[:, None, 1]], dim=2)
                # bottom_left = torch.stack([bboxes[:, None, 0], bboxes[:, None, 3]], dim=2)
                # bottom_right = torch.stack([bboxes[:, None, 2], bboxes[:, None, 3]], dim=2)
                # octagons = torch.cat([upper_right, upper_left, bottom_left, bottom_right], dim=1)
                # print('wrong')
                xmin, ymin = bboxes[:, 0], bboxes[:, 1]  # (n,)
                xmax, ymax = bboxes[:, 2], bboxes[:, 3]  # (n,)
                box = [xmax, ymin, xmin, ymin, xmin, ymax, xmax, ymax]
                box = torch.stack(box, dim=1).view(-1, 4, 2)
                octagons, edge_start_idx = self.uniform_upsample(
                    box[None], self.num_sampling
                )

                # just to suppress errors (DUMMY):
                init_box, _ = self.uniform_upsample(box[None], 40)
                ex_pts = init_box

            # List[np.array], element shape: (P, 2) OR None
            contours = self.get_simple_contour(gt_masks)

            # per instance
            # for (oct, cnt, w, h) in zip(octagons, contours, ws, hs):
            for (oct, cnt, in_box, ex_tar, w, h, s_idx) in zip(
                octagons, contours, init_box, ex_pts, ws, hs, edge_start_idx
            ):
                if cnt is None:
                    continue

                # used for normalization
                scale = torch.min(w, h)

                # make it clock-wise
                cnt = cnt[::-1] if Polygon(cnt).exterior.is_ccw else cnt
                """
                Quick fix for cityscapes
                """
                if Polygon(cnt).exterior.is_ccw:
                    continue

                assert not Polygon(
                    cnt
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                # sampling from octagon
                # oct_sampled_pts = self.uniform_sample(oct, self.num_sampling)
                #
                # oct_sampled_pts = oct_sampled_pts[::-1].copy() if Polygon(
                #     oct_sampled_pts).exterior.is_ccw else oct_sampled_pts
                oct_sampled_pts = oct.cpu().numpy()
                assert not Polygon(
                    oct_sampled_pts
                ).exterior.is_ccw, "1) contour must be clock-wise!"

                to_check = in_box.cpu().numpy()
                assert not Polygon(
                    to_check
                ).exterior.is_ccw, "0) init box must be clock-wise!"

                # if self.initial == 'box':
                #
                #     first_pt = aux_oct[0].cpu().numpy()
                #     tt_idx = np.argmin(np.power(oct_sampled_pts - first_pt, 2).sum(axis=1))
                #

                # sampling from ground truth
                oct_sampled_targets = self.uniform_sample(
                    cnt, len(cnt) * self.num_sampling * up_rate
                )  # (big, 2)
                # oct_sampled_targets = self.uniform_sample(cnt, len(cnt) * self.num_sampling * up_sample_rate)
                # i) find a single nearest, so that becomes ordered point sets
                tt_idx = np.argmin(
                    np.power(oct_sampled_targets - oct_sampled_pts[0], 2).sum(axis=1)
                )
                oct_sampled_targets = np.roll(oct_sampled_targets, -tt_idx, axis=0)[
                    :: len(cnt)
                ]

                if self.initial == "box" and self.new_matching:
                    oct_sampled_targets, aux_ext_idxs = get_aux_extreme_points(
                        oct_sampled_targets
                    )
                    tt_idx = np.argmin(
                        np.power(oct_sampled_pts - oct_sampled_targets[0], 2).sum(
                            axis=1
                        )
                    )
                    oct_sampled_pts = np.roll(oct_sampled_pts, -tt_idx, axis=0)
                    oct = torch.from_numpy(oct_sampled_pts).to(oct.device)
                    oct_sampled_targets = self.single_uniform_multisegment_matching(
                        oct_sampled_targets, oct_sampled_pts, aux_ext_idxs, up_rate
                    )
                    oct_sampled_targets = torch.tensor(
                        oct_sampled_targets, device=bboxes.device
                    ).float()
                else:
                    oct_sampled_targets = torch.tensor(
                        oct_sampled_targets, device=bboxes.device
                    )
                # assert not Polygon(oct_sampled_targets).exterior.is_ccw, '2) contour must be clock-wise!'

                # oct_sampled_pts = torch.tensor(oct_sampled_pts, device=bboxes.device)
                # dense_targets = torch.tensor(dense_targets, device=bboxes.device)

                oct_sampled_targets[..., 0].clamp_(min=0, max=img_size_per_im[1] - 1)
                oct_sampled_targets[..., 1].clamp_(min=0, max=img_size_per_im[0] - 1)

                dense_targets = oct_sampled_targets

                if self.initial == "octagon" and self.new_matching:

                    # oct_sampled_targets = self.single_segment_matching(dense_targets, oct, s_idx)

                    oct_sampled_targets = self.single_uniform_segment_matching(
                        dense_targets.cpu().numpy(),
                        oct.cpu().numpy(),
                        s_idx.cpu().numpy(),
                        up_rate,
                    )

                    oct_sampled_targets = torch.tensor(
                        oct_sampled_targets, device=bboxes.device
                    ).float()

                else:
                    # oct_sampled_targets = oct_sampled_targets[::up_rate]
                    pass

                if (
                    self.reorder_method == "curvegcn"
                ):  # TODO: but not applied to every time loss is calculated
                    oct_sampled_pts = torch.tensor(
                        oct_sampled_pts, device=oct_sampled_targets.device
                    )
                    oct_sampled_targets = self.reorder(
                        oct_sampled_targets, oct_sampled_pts
                    )

                # Jittering should happen after all the matching
                if self.jittering:
                    turbulence = (
                        torch.randn_like(ex_pts, device=bboxes.device) * self.jittering
                    )

                    # for box
                    ex_pts[:, 1::2, 0] += turbulence[:, 1::2, 0] * ws[:, None]
                    ex_pts[:, 0::2, 1] += turbulence[:, 0::2, 1] * hs[:, None]
                    # for ext
                    ex_pts[:, 0::2, 0] += turbulence[:, 0::2, 0] * ws[:, None] * 0.25
                    ex_pts[:, 1::2, 1] += turbulence[:, 1::2, 1] * hs[:, None] * 0.25

                    ex_pts[..., 0].clamp_(min=0, max=img_size_per_im[1] - 1)
                    ex_pts[..., 1].clamp_(min=0, max=img_size_per_im[0] - 1)

                poly_sample_locations.append(oct)
                dense_sample_targets.append(dense_targets)
                poly_sample_targets.append(oct_sampled_targets)
                image_index.append(im_i)
                scales.append(scale)
                whs.append([w, h])
                init_box_locs.append(in_box)
                init_ex_targets.append(ex_tar)

        init_ex_targets = torch.stack(init_ex_targets, dim=0)
        poly_sample_locations = torch.stack(poly_sample_locations, dim=0)
        init_box_locs = torch.stack(init_box_locs, dim=0)
        # init_ex_targets = torch.stack(init_ex_targets, dim=0)

        dense_sample_targets = torch.stack(dense_sample_targets, dim=0)
        poly_sample_targets = torch.stack(poly_sample_targets, dim=0)
        # edge_index = torch.stack(edge_index, dim=0)
        image_index = torch.tensor(image_index, device=bboxes.device)
        whs = torch.tensor(whs, device=bboxes.device)
        scales = torch.stack(scales, dim=0)

        # cls = torch.stack(cls, dim=0)
        return {
            "sample_locs": poly_sample_locations,
            "sample_targets": poly_sample_targets,
            "sample_dense_targets": dense_sample_targets,
            "scales": scales,
            "whs": whs,
            # "edge_idx": edge_index,
            "image_idx": image_index,
            "init_locs": init_box_locs,
            "init_targets": init_ex_targets,
        }

    @staticmethod
    def detect_second_extreme_pts(sampled_pts, ext_idx, dense_targets):
        band_factor = 0.03
        gap_factor = 0.05
        ch_pts = sampled_pts[ext_idx]
        top_down_y = ch_pts[0::2, 1]
        left_right_x = ch_pts[1::2, 0]
        pass

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

    def reorder(self, oct_sampled_targets, oct_sampled_pts):
        """
        :param oct_sampled_targets: (num_sampling x 2) for single instance
        :param oct_sampled_pts: same~
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

        # if (locations > 1).any() or (locations < -1).any():
        #     print('exceed grid sample boundary')
        #     if (locations > 1).any():
        #         print(locations[torch.where(locations>1)])
        #     else:
        #         print(locations[torch.where(locations < -1)])

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
            # TODO: After all almost fixed, try padding_mode='reflection' to see if there's improv
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

    def de_location(self, locations):
        # de-location (spatial relationship among locations; translation invariant)
        x_min = torch.min(locations[..., 0], dim=-1)[0]
        y_min = torch.min(locations[..., 1], dim=-1)[0]
        x_max = torch.max(locations[..., 0], dim=-1)[0]
        y_max = torch.max(locations[..., 1], dim=-1)[0]
        new_locations = locations.clone()
        if self.de_location_type == "derange":  # [0, 1]
            new_locations[..., 0] = (new_locations[..., 0] - x_min[..., None]) / (
                x_max[..., None] - x_min[..., None]
            )
            new_locations[..., 1] = (new_locations[..., 1] - y_min[..., None]) / (
                y_max[..., None] - y_min[..., None]
            )

        elif self.de_location_type == "demean":  # [-1, 1]
            new_locations[..., 0] = (
                2.0
                * (new_locations[..., 0] - x_min[..., None])
                / (x_max[..., None] - x_min[..., None])
                - 1.0
            )
            new_locations[..., 1] = (
                2.0
                * (new_locations[..., 1] - y_min[..., None])
                / (y_max[..., None] - y_min[..., None])
                - 1.0
            )
        elif self.de_location_type == "demin":
            new_locations[..., 0] = new_locations[..., 0] - x_min[..., None]
            new_locations[..., 1] = new_locations[..., 1] - y_min[..., None]
        else:
            raise ValueError("Invalid operation!", self.de_location_type)

        return new_locations

    @staticmethod
    def get_adj_ind(n_adj, n_nodes, device):
        ind = torch.LongTensor(
            [i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0]
        )
        ind = (torch.arange(n_nodes)[:, None] + ind[None]) % n_nodes
        return ind.to(device)

    def evolve(
        self,
        deformer,
        features,
        locations,
        image_idx,
        image_sizes,
        whs,
        att=False,
        path=None,
    ):

        locations_for_sample = locations if not self.detach else locations.detach()

        # with timer.env('snake_sample_feat'):
        sampled_features = self.get_locations_feature(
            features, locations_for_sample, image_idx
        )

        if self.attention:
            att_scores = sampled_features[:, :1, :]
            sampled_features = sampled_features[:, 1:, :]

        calibrated_locations = self.de_location(locations_for_sample)
        # calibrated_locations = self.de_location(locations).detach()
        # (Sigma{num_poly_i}, 2 + feat_dim, 128)
        concat_features = torch.cat(
            [sampled_features, calibrated_locations.permute(0, 2, 1)], dim=1
        )

        # if path is not None and self.track_path:
        #     concat_features = torch.cat([concat_features, path.permute(0, 2, 1)], dim=1)

        # if self.mark_index:
        #     pt_idx = torch.arange(self.num_sampling, device=concat_features.device).float() \
        #              / self.num_sampling
        #     pt_idx = pt_idx[None, None].repeat(concat_features.size(0), 1, 1)
        #     concat_features = torch.cat([concat_features, pt_idx], dim=1)

        # with timer.env('snake_evolves'):
        if self.conv_type == "ccn":
            pred_offsets = deformer(concat_features)

        elif self.conv_type == "gcn":
            adj = self.get_adj_ind(
                self.num_adj, concat_features.size(2), concat_features.device
            )
            pred_offsets = deformer(concat_features, adj)
        else:
            raise ValueError("Unsupported operation!")

        if self.pre_offset and isinstance(deformer, _PreOffSnakeNet):
            global_offset = pred_offsets[1].permute(0, 2, 1)
            pred_offsets = pred_offsets[0].permute(0, 2, 1) + global_offset
        else:
            pred_offsets = pred_offsets.permute(0, 2, 1)

        # if self.multi_offset > 1:
        #     pred_offsets = pred_offsets.reshape(-1, self.num_sampling, self.multi_offset, 2)
        #     dup_locations = locations.unsqueeze(-2).expand_as(pred_offsets)
        #
        #     # (n, num_sampling, multi, 2)
        #     multi_pred_locations = pred_offsets + dup_locations
        #
        #     if self.multi_offset == 3:
        #         l_preds = multi_pred_locations[:, :, 0, :]
        #         this_preds = multi_pred_locations[:, :, 1, :]
        #         r_preds = multi_pred_locations[:, :, 2, :]
        #         shifted_l = torch.cat([l_preds[:, -1:], l_preds[:, :-1]], dim=1)
        #         shifted_r = torch.cat([r_preds[:, 1:], r_preds[:, :1]], dim=1)
        #         pred_locations = (this_preds + shifted_l + shifted_r) / 3
        #         # pred_locations = locations + pred_offsets
        #         self.clip_locations(pred_locations, image_idx, image_sizes)
        #
        #         return pred_locations, multi_pred_locations
        #     else:
        #         raise ValueError("Not Supported!")

        if self.individual_scale:
            pred_offsets = torch.tanh(pred_offsets) * whs[:, None, :]

        if att:
            # print('att scores', att_scores)
            pred_offsets = pred_offsets * att_scores.permute(0, 2, 1)

        pred_locations = locations + pred_offsets

        self.clip_locations(pred_locations, image_idx, image_sizes)

        return pred_locations, None

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

    # def score(self, predictor, features, locations, image_idx, scales):
    #     # locations should be detached.
    #     if isinstance(locations, list):
    #         image_idx = image_idx.repeat(len(locations))
    #         locations = torch.cat(locations, dim=0)
    #     locations = locations.detach()
    #
    #     # TODO: Add perturbation later!!
    #
    #     sampled_features = self.get_locations_feature(features, locations, image_idx)
    #     calibrated_locations = self.de_location(locations)
    #     # (Sigma{num_poly_i}, 2 + feat_dim, 128)
    #     concat_features = torch.cat([sampled_features, calibrated_locations.permute(0, 2, 1)], dim=1)
    #
    #     # if self.mark_index:
    #     #     pt_idx = torch.arange(self.num_sampling, device=concat_features.device).float() \
    #     #              / self.num_sampling
    #     #     pt_idx = pt_idx[None, None].repeat(concat_features.size(0), 1, 1)
    #     #     concat_features = torch.cat([concat_features, pt_idx], dim=1)
    #
    #     pred_mask_scores = predictor(concat_features).squeeze()
    #     return pred_mask_scores, locations

    @staticmethod
    def contour_upsample_2x(locations):
        # (N, old_num, 2)
        new_locations = locations.new_empty(
            (locations.size(0), 2 * locations.size(1), locations.size(2))
        )
        new_locations[:, ::2, :] = locations
        shifted = torch.cat([locations[:, 1:, :], locations[:, :1, :]], dim=1)
        new_locations[:, 1::2, :] = (locations + shifted) / 2
        return new_locations

    def single_test(self, features, pred_instances):
        if self.original:
            with timer.env("init_octagon"):
                init_locs, image_idx = self.single_sample_quad_fast(pred_instances)
                if len(init_locs) == 0:
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
                init_pred_location, _ = self.evolve(
                    self.init_deformer, features, init_locs, image_idx, image_sizes, whs
                )
                pred_ex_pt = init_pred_location[:, ::10]  # expected to be (sum_n, 4, 2)

                locations, _ = self.get_octagon(pred_ex_pt, self.num_sampling)
        else:
            with timer.env("contour_sampling"):
                if self.initial == "octagon":
                    locations, image_idx = self.sample_octagons_fast(pred_instances)
                    image_idx = None
                    # raise ValueError('no octagon pls')
                elif self.initial == "box":
                    locations, image_idx = self.single_sample_bboxes_fast(
                        pred_instances
                    )
                else:
                    raise ValueError("Invalid initial!")

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

        if self.attention:
            edge_band = features[:, :1, ...]
            features = features[:, 1:, ...]

        with timer.env("snake_feat_prep"):
            if self.coord_conv[0]:
                features = self.add_coords(features)
            for i in range(self.num_convs):
                features = self.bottom_out[i](features)

        with timer.env("snake_deformation"):
            if self.attention:
                features = torch.cat([edge_band, features], dim=1)
                location_preds = []

                for i in range(len(self.num_iter)):
                    deformer = self.__getattr__("deformer" + str(i))
                    if i == 0:
                        pred_location, _ = self.evolve(
                            deformer, features, locations, image_idx, image_sizes, whs
                        )
                    else:
                        pred_location, _ = self.evolve(
                            deformer,
                            features,
                            pred_location,
                            image_idx,
                            image_sizes,
                            whs,
                            att=True,
                        )
                    location_preds.append(pred_location)
            else:
                location_preds = []
                for i in range(len(self.num_iter)):
                    deformer = self.__getattr__("deformer" + str(i))
                    if i == 0:
                        pred_location, _ = self.evolve(
                            deformer, features, locations, image_idx, image_sizes, whs
                        )
                    else:
                        pred_location, _ = self.evolve(
                            deformer,
                            features,
                            pred_location,
                            image_idx,
                            image_sizes,
                            whs,
                        )

                    location_preds.append(pred_location)

        with timer.env("snake_postprocess"):
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

            locations, reg_targets, scales, image_idx = (
                training_targets["sample_locs"],
                training_targets["sample_targets"],
                training_targets["scales"],
                training_targets["image_idx"],
            )
            init_locs, init_targets = (
                training_targets["init_locs"],
                training_targets["init_targets"],
            )
            # classes = training_targets["classes"]
            # dense_targets = training_targets["sample_dense_targets"]
            whs = training_targets["whs"]
        else:
            if not self.visualize_path:
                return self.single_test(features, pred_instances)

            with timer.env("snake_sample"):
                assert pred_instances is not None

                if self.original:
                    with timer.env("init_octagon"):
                        init_locs, image_idx = self.sample_quad_fast(pred_instances)
                        if len(init_locs) == 0:
                            return pred_instances, {}
                        image_sizes = list(map(lambda x: x.image_size, pred_instances))
                        # print(image_sizes)
                        # print(features.shape)
                        # bboxes = pred_instances[0].pred_boxes.tensor
                        # print(bboxes.max(0)[0], bboxes.min(0)[0])
                        pred_instances[0].pred_boxes.clip(image_sizes)
                        bboxes = pred_instances[0].pred_boxes.tensor
                        ws = bboxes[:, 2] - bboxes[:, 0]
                        hs = bboxes[:, 3] - bboxes[:, 1]
                        whs = torch.stack([ws, hs], dim=1)
                        init_pred_location, _ = self.evolve(
                            self.init_deformer,
                            features,
                            init_locs,
                            image_idx,
                            image_sizes,
                            whs,
                        )
                        pred_ex_pt = init_pred_location[
                            :, ::10
                        ]  # expected to be (sum_n, 4, 2)

                        locations, _ = self.get_octagon(pred_ex_pt, self.num_sampling)
                else:
                    if self.initial == "octagon":
                        locations, image_idx = self.sample_octagons_fast(pred_instances)
                    elif self.initial == "box":
                        locations, image_idx = self.sample_bboxes_fast(pred_instances)
                    else:
                        raise ValueError("Invalid initial!")
                    if len(locations) == 0:
                        return pred_instances, {}
                    image_sizes = list(map(lambda x: x.image_size, pred_instances))
                    bboxes = list(map(lambda x: x.pred_boxes.tensor, pred_instances))
                    bboxes = cat(bboxes, dim=0)

                    ws = bboxes[:, 2] - bboxes[:, 0]
                    hs = bboxes[:, 3] - bboxes[:, 1]
                    whs = torch.stack([ws, hs], dim=1)
                # image_idx = None

        if self.selective_refine:
            edge_band = features[:, :1, ...]
            if not (self.pred_edge and not self.training):
                features = features[:, 1:, ...]

        if self.attention:
            edge_band = features[:, :1, ...]
            features = features[:, 1:, ...]

        # init the octagons
        if self.original and self.training:
            init_pred_location, _ = self.evolve(
                self.init_deformer, features, init_locs, image_idx, image_sizes, whs
            )

        # feature preparation
        preserved_location_pred = None
        if not (self.use_aspp or self.use_psp):
            if self.structure == "sequential":
                with timer.env("snake_feat_prep"):
                    if self.coord_conv[0]:
                        features = self.add_coords(features)
                    for i in range(self.num_convs):
                        features = self.bottom_out[i](features)

                if self.attention:
                    features = torch.cat([edge_band, features], dim=1)
                    location_preds = []
                    multi_location_preds = []
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
                        multi_location_preds.append(multi_pred_locations)
                else:
                    location_preds = []
                    multi_location_preds = []
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
                            if self.track_path:
                                path = (pred_location - locations).detach()
                            else:
                                path = None
                        else:
                            if self.last_up_sample and i == len(self.num_iter) - 1:
                                pred_location = self.contour_upsample_2x(pred_location)
                            pred_location, multi_pred_locations = self.evolve(
                                deformer,
                                features,
                                pred_location,
                                image_idx,
                                image_sizes,
                                whs,
                                path,
                            )
                            if self.track_path:
                                path = (pred_location - location_preds[-1]).detach()
                            else:
                                path = None

                        location_preds.append(pred_location)

                        if self.double_selective_refine and i == 1:
                            with timer.env("snake_postprocess"):
                                sampled_edgeness = self.get_locations_feature(
                                    edge_band, location_preds[-2], image_idx
                                )
                                point_offedge_score = 1 - sampled_edgeness.permute(
                                    0, 2, 1
                                )
                                last_offsets = location_preds[-1] - location_preds[-2]
                                modified_last_offsets = (
                                    last_offsets * point_offedge_score
                                )
                                pred_location = (
                                    location_preds[-2] + modified_last_offsets
                                )
                                if not self.training:
                                    location_preds[-1] = pred_location
                                preserved_location_pred = pred_location

                        multi_location_preds.append(multi_pred_locations)

            elif self.structure == "parallel":
                # if self.num_iter != self.num_convs, "No one-2-one matching between stage and branch."
                para_features = {}
                location_preds = []
                multi_location_preds = []

                with timer.env("snake_feat_prep"):
                    for i in range(self.num_convs):
                        feature_name = "feat" + str(i)
                        if self.coord_conv[i]:
                            tmp_features = self.add_coords(features)
                        else:
                            tmp_features = features

                        if self.edge_in_separate[i]:
                            tmp_features = torch.cat([edge_band, tmp_features], dim=1)

                        para_features[feature_name] = self.bottom_out[i](tmp_features)

                for i in range(len(self.num_iter)):
                    get_name = "feat" + str(self.num_iter[i])
                    deformer = self.__getattr__("deformer" + str(i))
                    if i == 0:
                        pred_location, multi_pred_locations = self.evolve(
                            deformer,
                            para_features[get_name],
                            locations,
                            image_idx,
                            image_sizes,
                            whs,
                        )
                        if self.track_path:
                            path = pred_location - locations
                        else:
                            path = None
                    else:
                        if self.last_up_sample and i == len(self.num_iter) - 1:
                            pred_location = self.contour_upsample_2x(pred_location)
                        pred_location, multi_pred_locations = self.evolve(
                            deformer,
                            para_features[get_name],
                            pred_location,
                            image_idx,
                            image_sizes,
                            whs,
                            path,
                        )
                        if self.track_path:
                            path = pred_location - location_preds[-1]
                        else:
                            path = None
                    location_preds.append(pred_location)
                    multi_location_preds.append(multi_pred_locations)
            else:
                raise ValueError("Invalid structure!")
        else:
            h, w = features.size(2), features.size(3)
            with timer.env("snake_feat_prep"):
                if self.use_aspp:
                    image_fea = F.interpolate(
                        input=self.image_pool(features),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    encoder_fea = [aspp_stage(features) for aspp_stage in self.aspp]
                    encoder_fea.append(image_fea)
                    encoder_fea = self.encode_fea(torch.cat(encoder_fea, dim=1))
                elif self.use_psp:
                    priors = [
                        F.interpolate(
                            input=stage(features),
                            size=(h, w),
                            mode="bilinear",
                            align_corners=False,
                        )
                        for stage in self.psp
                    ]
                    priors.append(features)
                    encoder_fea = self.relu(self.encode_fea(torch.cat(priors, 1)))
                else:
                    raise ValueError("Invalid feature fusion!")

            location_preds = []
            multi_location_preds = []

            for i in range(len(self.num_iter)):
                deformer = self.__getattr__("deformer" + str(i))
                if i == 0:
                    pred_location, multi_pred_locations = self.evolve(
                        deformer, encoder_fea, locations, image_idx, image_sizes, whs
                    )
                    if self.track_path:
                        path = (pred_location - locations).detach()
                    else:
                        path = None
                else:
                    if self.last_up_sample and i == len(self.num_iter) - 1:
                        pred_location = self.contour_upsample_2x(pred_location)
                    pred_location, multi_pred_locations = self.evolve(
                        deformer,
                        encoder_fea,
                        pred_location,
                        image_idx,
                        image_sizes,
                        whs,
                        path,
                    )
                    if self.track_path:
                        path = (pred_location - location_preds[-1]).detach()
                    else:
                        path = None
                location_preds.append(pred_location)
                multi_location_preds.append(multi_pred_locations)

        # if self.ms_head is not None:
        #     if self.training:
        #         tmp_locations = location_preds.copy()
        #         tmp_locations.append(locations)
        #     else:
        #         tmp_locations = location_preds[-1]
        #     ms_feat = self.ms_layer(features)
        #     mask_scores, perturbed_locations = self.score(self.ms_head, ms_feat, tmp_locations, image_idx)
        # else:
        #     mask_scores = None

        if self.training:
            loss = {}
            # if self.ms_head is not None:
            #     loss['loss_ms'] = self.compute_loss_for_maskious(
            #         classes,
            #         reg_targets,
            #         perturbed_locations,
            #         mask_scores) * self.ms_loss_weight
            if self.original:
                point_weight = (
                    torch.tensor(1, device=scales.device).float()
                    / scales[:, None, None]
                )
                loss_init_pred = self.loss_reg(
                    init_pred_location[:, ::10] * point_weight,
                    init_targets * point_weight,
                )
                loss["loss_init"] = loss_init_pred * self.refine_loss_weight * 0.5

            for i, (pred, multi_pred) in enumerate(
                zip(location_preds, multi_location_preds)
            ):
                loss_name = "loss_stage_" + str(i)
                stage_weight = (
                    self.loss_distribution[i] if self.stage_loss_weigh else 1 / 3
                )
                loss_func = self.loss_reg
                if i == 2 and self.loss_refine is not None:
                    loss_func = self.loss_refine

                if self.loss_adaptive:
                    if self.last_up_sample and i == 2:
                        reg_targets = self.contour_upsample_2x(reg_targets)
                    dynamic_reg_targets = self.reorder_perloss(
                        reg_targets, pred.detach()
                    )
                else:
                    dynamic_reg_targets = reg_targets

                if self.point_loss_weight:
                    point_weight = self.normalize_point_diffs(
                        dynamic_reg_targets, pred.detach()
                    )
                else:
                    point_weight = (
                        torch.tensor(1, device=scales.device).float()
                        / scales[:, None, None]
                    )
                    if self.individual_scale:
                        point_weight = (
                            torch.tensor(1, device=scales.device).float()
                            / whs[:, None, :]
                        )

                # stage_loss = loss_func(pred * point_weight, dynamic_reg_targets * point_weight) * stage_weight
                if (
                    i == len(self.num_iter) - 1
                    and self.last_chamfer
                    and self.loss_neighbor is not None
                ):
                    # # pred: (n, 128, 2)
                    # neighbor1 = torch.cat([pred[:, 1:], pred[:, :1]], dim=1) * point_weight
                    # neighbor2 = torch.cat([pred[:, -1:], pred[:, :-1]], dim=1) * point_weight
                    # l1 = (pred * point_weight - neighbor1).abs()
                    # l2 = (pred * point_weight - neighbor2).abs()
                    # neighbor_loss = self.loss_neighbor(l1, l2, )
                    # loss['loss_neighbor'] = neighbor_loss * 1000
                    #
                    # print(neighbor_loss)

                    neighbor1 = torch.cat([pred[:, 1:], pred[:, :1]], dim=1)

                    l1 = ((pred - neighbor1).pow(2).sum(2) + 1e-6).sqrt()

                    b = l1.size(0)
                    n = l1.size(1)
                    mean_len = l1.mean(1)
                    neighbor_loss = (
                        (l1 - mean_len[:, None]).pow(2).sum(1) / n + 1e-6
                    ).sqrt().sum() / b
                    loss["loss_deviation"] = neighbor_loss * 0.2

                if i == len(self.num_iter) - 2 and self.double_selective_refine:
                    sample_locations = preserved_location_pred.detach()
                    sampled_edgeness = self.get_locations_feature(
                        edge_band, sample_locations, image_idx
                    )
                    point_offedge_score = 1 - sampled_edgeness.permute(0, 2, 1).detach()
                    stage_loss = (
                        loss_func(
                            pred * point_weight,
                            dynamic_reg_targets * point_weight,
                            weight=point_offedge_score,
                        )
                        * stage_weight
                    )
                    stage_loss *= (
                        point_offedge_score.size(0)
                        * point_offedge_score.size(1)
                        / point_offedge_score.sum()
                    )
                elif i == len(self.num_iter) - 1 and self.selective_refine:
                    sample_locations = (
                        self.contour_upsample_2x(location_preds[-2].detach())
                        if self.last_up_sample
                        else location_preds[-2].detach()
                    )
                    sampled_edgeness = self.get_locations_feature(
                        edge_band, sample_locations, image_idx
                    )
                    # TODO: previous BUG! Used octagon locs...
                    # TODO: Really need to be careful!
                    # TODO: BUG twice here... 1) oct locs; 2) last locs
                    # TODO: BUT should be second last!!!
                    point_offedge_score = 1 - sampled_edgeness.permute(0, 2, 1).detach()
                    stage_loss = (
                        loss_func(
                            pred * point_weight,
                            dynamic_reg_targets * point_weight,
                            weight=point_offedge_score,
                        )
                        * stage_weight
                    )
                    stage_loss *= (
                        point_offedge_score.size(0)
                        * point_offedge_score.size(1)
                        / point_offedge_score.sum()
                    )

                    if self.loss_neighbor is not None:
                        neighbor_loss = self.loss_neighbor(
                            l1, l2, weight=point_offedge_score
                        )
                        loss["loss_neighbor"] = neighbor_loss * 1000
                else:
                    if self.multi_offset > 1:
                        l_preds = multi_pred[:, :, 0, :]
                        this_preds = multi_pred[:, :, 1, :]
                        r_preds = multi_pred[:, :, 2, :]
                        shifted_l = torch.cat([l_preds[:, -1:], l_preds[:, :-1]], dim=1)
                        shifted_r = torch.cat([r_preds[:, 1:], r_preds[:, :1]], dim=1)

                        this_loss = (
                            loss_func(
                                this_preds * point_weight,
                                dynamic_reg_targets * point_weight,
                            )
                            * stage_weight
                        )
                        l_loss = (
                            loss_func(
                                shifted_l * point_weight,
                                dynamic_reg_targets * point_weight,
                            )
                            * stage_weight
                        )
                        r_loss = (
                            loss_func(
                                shifted_r * point_weight,
                                dynamic_reg_targets * point_weight,
                            )
                            * stage_weight
                        )
                        avg_loss = (
                            loss_func(
                                pred * point_weight, dynamic_reg_targets * point_weight
                            )
                            * stage_weight
                        )

                        stage_loss = (this_loss + l_loss + r_loss + avg_loss) / 4

                    else:
                        stage_loss = (
                            loss_func(
                                pred * point_weight, dynamic_reg_targets * point_weight
                            )
                            * stage_weight
                        )

                loss[loss_name] = stage_loss * self.refine_loss_weight

            # print(time.time() - start)
            return [], loss
        else:
            print("here")
            print(self.visualize_path)
            # TODO: speed up this to use a single get_locations_feature
            with timer.env("snake_postprocess"):
                if self.selective_refine:
                    sampled_edgeness = self.get_locations_feature(
                        edge_band, location_preds[-2], image_idx
                    )
                    point_offedge_score = 1 - sampled_edgeness.permute(0, 2, 1)
                    last_offsets = location_preds[-1] - location_preds[-2]
                    modified_last_offsets = last_offsets * point_offedge_score
                    location_preds[-1] = location_preds[-2] + modified_last_offsets

                new_instances = self.predict_postprocess(
                    pred_instances,
                    locations,
                    location_preds,
                    multi_location_preds,
                    image_idx,
                )
            return new_instances, {}

    @staticmethod
    def normalize_point_diffs(targets, preds):
        """
        Adaptively adjust the penalty, concept-wise the loss is much more reasonable.
        :param targets: (\sum{k}, num_sampling, 2) for all instances
        :param preds: same~
        :return:
        """
        dist = (targets - preds).pow(2).sum(2)
        point_weight = dist / torch.max(dist, dim=1)[0][:, None]
        return point_weight[..., None]

    def predict_postprocess(
        self,
        pred_instances,
        locations,
        location_preds,
        multi_location_preds,
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

            multi_pred_per_im = multi_location_preds[-1]
            if multi_pred_per_im is not None:
                instance_per_im.last_multi_pred = multi_pred_per_im[image_idx == i]

            if scores is not None:
                score_per_im = scores[image_idx == i]
                instance_per_im.scores *= score_per_im
                # TODO: multiplication with associated cls.

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
                print("add in pred_path!")

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
