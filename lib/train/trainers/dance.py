import torch.nn as nn
from lib.utils import net_utils
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.ct_crit = net_utils.FocalLoss()
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.reg_crit = net_utils.IndL1Loss1d('smooth_l1')
        # self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss
        self.edge_crit = net_utils.DiceLoss(0, 255)

    def forward(self, batch):
        output = self.net(batch['inp'], batch)

        scalar_stats = {}
        loss = 0

        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']),
                               batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'],
                               batch['ct_01'])
        scalar_stats.update({'wh_loss': wh_loss})
        loss += 0.1 * wh_loss

        # reg_loss = self.reg_crit(output['reg'], batch['reg'], batch['ct_ind'], batch['ct_01'])
        # scalar_stats.update({'reg_loss': reg_loss})
        # loss += reg_loss

        # ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
        # scalar_stats.update({'ex_loss': ex_loss})
        # loss += ex_loss

        # py_loss = 0
        # output['py_pred'] = [output['py_pred'][-1]]
        # for i in range(len(output['py_pred'])):
        #     py_loss += self.py_crit(output['py_pred'][i], output['i_gt_py']) / len(output['py_pred'])
        # scalar_stats.update({'py_loss': py_loss})
        # loss += py_loss

        # dance losses
        edge_loss = self.edge_crit(output['pred_edge_full'], batch['edge_map'])
        scalar_stats.update({'edge_loss': edge_loss})
        loss += edge_loss

        py_loss_all = 0
        # 1) HAS individual scaling
        # point_weight = 1
        # print(output['py_pred'][0].shape)
        # print(batch['whs'].shape)
        # point_weight = torch.tensor(
        #     1, device=output['py_pred'][0].device).float() / (
        #         output['batched_whs'][:, None, :] * snake_config.ro)

        # for i in range(len(output['py_pred'])):
        #     py_loss_all += self.py_crit(output['py_pred'][i] * point_weight,
        #                                 output['i_gt_py'] * point_weight)
        # py_loss_all *= 250
        # scalar_stats.update({'py_loss': py_loss_all})
        # loss += py_loss_all

        # 2) NO individual scaling
        point_weight = 1
        for i in range(len(output['py_pred'])):
            py_loss_all += self.py_crit(
                output['py_pred'][i] * point_weight,
                output['i_gt_py'] * point_weight) / len(output['py_pred'])
        scalar_stats.update({'py_loss': py_loss_all})
        loss += py_loss_all

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
