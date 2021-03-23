import torch
import torch.nn as nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from lib.utils.dance import dance_config, dance_gcn_utils, dance_decode
from lib.utils.snake import snake_config, snake_decode

from .snake import SnakeNet
from .wrappers import Conv2d


class Dance(nn.Module):
    def __init__(self):
        super(Dance, self).__init__()
        # pre_dance
        pre_dance_conv_dims = dance_config.pre_dance_conv_dims
        self.bottom_out = nn.ModuleList()
        for i in range(2):
            norm_module = nn.GroupNorm(32, pre_dance_conv_dims)
            if i == 0:
                dim_in = 64
            else:
                dim_in = pre_dance_conv_dims
            my_conv = Conv2d(
                dim_in,
                pre_dance_conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=False,
                norm=norm_module,
                activation=F.relu,
            )
            self.bottom_out.append(my_conv)

        # dance_head: snakes
        self.iter = 3
        for i in range(self.iter):
            evolve_gcn = SnakeNet(state_dim=128,
                                  edge_feature_dim=pre_dance_conv_dims + 2)
            self.__setattr__('deformer' + str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # such order so that we have proper init

        # edge
        edge_conv_dims = dance_config.edge_conv_dims
        conv = Conv2d(
            64,  # dla feature output channel
            edge_conv_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.GroupNorm(32, edge_conv_dims),
            activation=F.relu,
        )
        pred = Conv2d(edge_conv_dims, 1, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(conv)
        weight_init.c2_msra_fill(pred)
        self.edge_predictor = nn.Sequential(conv, pred)

        # attention
        conv1 = Conv2d(
            1,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.GroupNorm(4, 32),
            activation=F.relu,
        )
        conv2 = Conv2d(
            32,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.GroupNorm(4, 32),
            activation=F.relu,
        )
        conv3 = Conv2d(
            32,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            activation=torch.sigmoid,
        )
        weight_init.c2_msra_fill(conv1)
        weight_init.c2_msra_fill(conv2)
        nn.init.normal_(conv3.weight, 0, 0.01)
        nn.init.constant_(conv3.bias, 0)
        self.attender = nn.Sequential(conv1, conv2, conv3)

    def prepare_training(self, output, batch):
        # batch all the polys && label the their indices (to each image)
        init = dance_gcn_utils.prepare_training(output, batch)
        output.update({
            'init_box': init['init_box'],
            'targ_poly': init['targ_poly']
        })
        return init

    def prepare_training_evolve(self, output, batch, init):
        ct_num = batch['meta']['ct_num'].sum()
        evolve = dance_gcn_utils.prepare_training_evolve(
            output['ex_pred'], init, ct_num)
        output.update({
            'i_it_py': evolve['i_it_py'],
            'c_it_py': evolve['c_it_py'],
            'i_gt_py': evolve['i_gt_py']
        })
        evolve.update({'ind': init['ind'][:evolve['i_gt_py'].size(0)]})
        return evolve

    # def prepare_testing_init(self, output):
    #     i_it_4py = snake_decode.get_init(output['cp_box'][None])
    #     i_it_4py = dance_gcn_utils.uniform_upsample(i_it_4py,
    #                                                 snake_config.init_poly_num)
    #     c_it_4py = dance_gcn_utils.img_poly_to_can_poly(i_it_4py)

    #     i_it_4py = i_it_4py[0]
    #     c_it_4py = c_it_4py[0]
    #     ind = output['roi_ind'][output['cp_ind'].long()]
    #     init = {'i_it_4py': i_it_4py, 'c_it_4py': c_it_4py, 'ind': ind}
    #     output.update({'it_ex': init['i_it_4py']})

    #     return init

    def prepare_testing_locations(self, output):
        # if len(output['cp_box']) == 0:
        # print(output['detection'].shape)

        box = output['detection'][..., :4]
        score = output['detection'][..., 4]

        ind = score > snake_config.ct_score
        # print('ind', ind.shape)
        i_it_4py = dance_decode.get_init(box)
        # print('i_it_4py', i_it_4py.shape)
        i_it_4py = i_it_4py[ind]
        if len(i_it_4py) == 0:
            init = {
                'i_it_4py':
                i_it_4py,
                'ind':
                torch.cat([
                    torch.full([ind[i].sum()], i) for i in range(ind.size(0))
                ],
                          dim=0)
            }
            output.update({'it_location': init['i_it_4py']})
            tmp = {'boxes': i_it_4py}
            init.update(tmp)
            return init

        i_it_4py = i_it_4py[None]

        # print('i_it_4py after', i_it_4py.shape)

        tmp = {'boxes': i_it_4py[0]}
        # print(i_it_4py.shape)
        i_it_4py = dance_gcn_utils.uniform_upsample(i_it_4py,
                                                    snake_config.poly_num)
        # c_it_4py = dance_gcn_utils.img_poly_to_can_poly(i_it_4py)

        i_it_4py = i_it_4py[0]
        # c_it_4py = c_it_4py[0]
        ind = torch.cat(
            [torch.full([ind[i].sum()], i) for i in range(ind.size(0))], dim=0)
        # print(ind)
        init = {'i_it_4py': i_it_4py, 'ind': ind}
        init.update(tmp)
        output.update({'it_location': init['i_it_4py']})

        output['detection'] = output['detection'][output['detection'][..., 4] > snake_config.ct_score]

        return init

    def prepare_testing_evolve(self, output, h, w):
        ex = output['ex']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w - 1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h - 1)
        evolve = dance_gcn_utils.prepare_testing_evolve(ex)
        output.update({'it_py': evolve['i_it_py']})
        return evolve

    # def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
    #     if len(i_it_poly) == 0:
    #         return torch.zeros([0, 4, 2]).to(i_it_poly)

    #     h, w = cnn_feature.size(2), cnn_feature.size(3)
    #     init_feature = dance_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly,
    #                                                    ind, h, w)
    #     center = (torch.min(i_it_poly, dim=1)[0] +
    #               torch.max(i_it_poly, dim=1)[0]) * 0.5
    #     ct_feature = dance_gcn_utils.get_gcn_feature(cnn_feature, center[:,
    #                                                                      None],
    #                                                  ind, h, w)
    #     init_feature = torch.cat(
    #         [init_feature, ct_feature.expand_as(init_feature)], dim=1)
    #     init_feature = self.fuse(init_feature)

    #     init_input = torch.cat(
    #         [init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
    #     adj = dance_gcn_utils.get_adj_ind(snake_config.adj_num,
    #                                       init_input.size(2),
    #                                       init_input.device)
    #     i_poly = i_it_poly + snake(init_input, adj).permute(0, 2, 1)
    #     i_poly = i_poly[:, ::snake_config.init_poly_num // 4]

    #     return i_poly

    def de_location(self, locations):
        # de-location (spatial relationship among locations; translation invariant)
        x_min = torch.min(locations[..., 0], dim=-1)[0]
        y_min = torch.min(locations[..., 1], dim=-1)[0]
        x_max = torch.max(locations[..., 0], dim=-1)[0]
        y_max = torch.max(locations[..., 1], dim=-1)[0]
        new_locations = locations.clone()

        new_locations[..., 0] = (new_locations[..., 0] - x_min[..., None]) / \
                                (x_max[..., None] - x_min[..., None])
        new_locations[..., 1] = (new_locations[..., 1] - y_min[..., None]) / \
                                (y_max[..., None] - y_min[..., None])
        return new_locations

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = dance_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly,
                                                       ind, h, w)
        c_it_poly = c_it_poly * snake_config.ro
        init_input = torch.cat(
            [init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = dance_gcn_utils.get_adj_ind(snake_config.adj_num,
                                          init_input.size(2),
                                          init_input.device)
        i_poly = i_it_poly * snake_config.ro + snake(init_input, adj).permute(
            0, 2, 1)
        return i_poly

    def evolve(self, deformer, cnn_feature, locations, ind, whs, att=False):
        if len(locations) == 0:
            return torch.zeros_like(locations)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        sampled_features = dance_gcn_utils.get_gcn_feature(
            cnn_feature, locations, ind, h, w)

        # extract attention scores
        att_scores = sampled_features[:, :1, :]
        sampled_features = sampled_features[:, 1:, :]

        calibrated_locations = self.de_location(
            locations * snake_config.ro)  # original scaled locations

        # calibrated_locations = self.de_location(locations).detach()

        # (Sigma{num_poly_i}, 2 + feat_dim, 128)
        concat_features = torch.cat(
            [sampled_features,
             calibrated_locations.permute(0, 2, 1)], dim=1)

        pred_offsets = deformer(
            concat_features)  # should act on original scales
        pred_offsets = pred_offsets.permute(0, 2, 1)

        # better normalization
        # print(pred_offsets.shape)
        # print(whs[:, None, :].shape)
        # {k, 128, 2}

        # no scaling:
        # pred_offsets = torch.tanh(pred_offsets) * (whs[:, None, :] *
        #                                            snake_config.ro)

        # modulated by edge attention
        pred_offsets = pred_offsets * att_scores.permute(0, 2, 1)

        pred_locations = locations * snake_config.ro + pred_offsets

        # TODO: consider add this clip
        # self.clip_locations(pred_locations, image_idx, image_sizes)

        return pred_locations

    def forward(self, output, cnn_feature, batch=None):
        ret = output

        # Attentive
        pred_logits = self.edge_predictor(cnn_feature)
        pred_edge = pred_logits.sigmoid()
        att_map = self.attender(1 - pred_edge)

        if batch is not None and 'test' not in batch['meta']:
            pred_edge_full = F.interpolate(pred_edge,
                                           scale_factor=snake_config.ro,
                                           mode="bilinear",
                                           align_corners=False)
            ret.update({'pred_edge_full': pred_edge_full})

            features = cnn_feature
            for i in range(2):
                features = self.bottom_out[i](features)
            features = torch.cat([att_map, features], dim=1)

            # prepare deformation data

            ct_01 = batch['ct_01'].byte()
            locations = dance_gcn_utils.collect_training(
                batch['init_box'], ct_01)  # 1/4 scale
            targ_poly = dance_gcn_utils.collect_training(
                batch['targ_poly'], ct_01)  # 1/4 scale
            whs = dance_gcn_utils.collect_training(batch['whs'], ct_01)
            ret.update({'batched_whs': whs})
            ct_num = batch['meta']['ct_num']
            poly_ind = torch.cat(
                [torch.full([ct_num[i]], i) for i in range(ct_01.size(0))],
                dim=0)

            location_preds = []
            for i in range(self.iter):
                deformer = self.__getattr__('deformer' + str(i))
                if i == 0:
                    pred_location = self.evolve(deformer, features, locations,
                                                poly_ind, whs)
                else:
                    pred_location = pred_location / snake_config.ro
                    pred_location = self.evolve(deformer,
                                                features,
                                                pred_location,
                                                poly_ind,
                                                whs,
                                                att=True)
                location_preds.append(pred_location)

            ret.update({
                'py_pred': location_preds,
                'i_gt_py': targ_poly * snake_config.ro
            })

        if not self.training:
            with torch.no_grad():
                init = self.prepare_testing_locations(output)
                img_h, img_w = snake_config.ro * cnn_feature.shape[
                    2], snake_config.ro * cnn_feature.shape[3]
                locations = output['it_location']
                boxes = init['boxes']
                # print('loca', locations.shape)
                boxes[:, 0].clamp_(min=0, max=img_w)
                boxes[:, 1].clamp_(min=0, max=img_h)
                boxes[:, 2].clamp_(min=0, max=img_w)
                boxes[:, 3].clamp_(min=0, max=img_h)
                ws = boxes[:, 2] - boxes[:, 0]
                hs = boxes[:, 3] - boxes[:, 1]
                whs = torch.stack(
                    [ws, hs],
                    dim=1)  # whs SHOULD BE 1/4 SCALE (same as training)
                # print('whs', whs.shape)

                features = cnn_feature
                for i in range(2):
                    features = self.bottom_out[i](features)
                features = torch.cat([att_map, features], dim=1)

                pys = []
                for i in range(self.iter):
                    deformer = self.__getattr__('deformer' + str(i))
                    if i == 0:
                        pred_location = self.evolve(deformer, features,
                                                    locations, init['ind'],
                                                    whs)
                    else:
                        pred_location = pred_location / snake_config.ro
                        pred_location = self.evolve(deformer,
                                                    features,
                                                    pred_location,
                                                    init['ind'],
                                                    whs,
                                                    att=True)
                    pys.append(pred_location / snake_config.ro)

                # ex = self.init_poly(self.init_gcn, cnn_feature,
                #                     init['i_it_4py'], init['c_it_4py'],
                #                     init['ind'])
                # ret.update({'ex': ex})

                # evolve = self.prepare_testing_evolve(output,
                #                                      cnn_feature.size(2),
                #                                      cnn_feature.size(3))
                # py = self.evolve_poly(self.evolve_gcn, cnn_feature,
                #                       evolve['i_it_py'], evolve['c_it_py'],
                #                       init['ind'])
                # pys = [py / snake_config.ro]
                # for i in range(self.iter):
                #     py = py / snake_config.ro
                #     c_py = dance_gcn_utils.img_poly_to_can_poly(py)
                #     evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                #     py = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py,
                #                           init['ind'])
                #     pys.append(py / snake_config.ro)
                ret.update({'py': pys})

                # print(type(pys[-1]))
                # print(pys[-1].shape)

        return output
