import math
from os.path import join as path_join

import cv2
import numpy as np
import torch.utils.data as data

from lib.utils import data_utils
from lib.utils.dance import (dance_cityscapes_utils, dance_config,
                             visualize_utils)
from lib.utils.rcnn_snake import rcnn_snake_config as snake_config
from shapely.geometry import Polygon

# import logging
# logger = logging.getLogger("RCNNDance")


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.anns = np.array(dance_cityscapes_utils.read_dataset(ann_file)[:])
        self.anns = self.anns[:10] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = dance_cityscapes_utils.JSON_DICT

        self.edge_map_root = '/'.join(
            data_root.split('/')[:-1] + ['edge_' + split])

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [np.array(obj['components']) for obj in anno]
        cls_ids = [
            self.json_category_id_to_contiguous_id[obj['label']]
            for obj in anno
        ]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width,
                                trans_output, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [np.array(poly['poly']) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = dance_cityscapes_utils.transform_polys(
                polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys):
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            polys = dance_cityscapes_utils.filter_tiny_polys(instance)
            polys = dance_cityscapes_utils.get_cw_polys(polys)
            polys = [
                poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])]
                for poly in polys
            ]
            instance_polys_.append(polys)
        return instance_polys_

    def get_amodal_boxes(self, extreme_points):
        boxes = []
        for instance_points in extreme_points:
            if len(instance_points) == 0:
                box = []
            else:
                instance = np.concatenate(instance_points)
                box = np.concatenate(
                    [np.min(instance, axis=0),
                     np.max(instance, axis=0)])
            boxes.append(box)
        return boxes

    def get_extreme_points(self, instance_polys):
        extreme_points = []
        for instance in instance_polys:
            points = [
                dance_cityscapes_utils.get_extreme_points(poly)
                for poly in instance
            ]
            extreme_points.append(points)
        return extreme_points

    def prepare_adet(self, box, ct_hm, cls_id, wh, ct_ind):
        if len(box) == 0:
            return

        ct_hm = ct_hm[cls_id]

        x_min, y_min, x_max, y_max = box
        ct = np.round([(x_min + x_max) / 2,
                       (y_min + y_max) / 2]).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

    def prepare_rcnn(self, abox, instance, cp_hm, cp_wh, cp_ind):
        if len(abox) == 0:
            return

        x_min, y_min, x_max, y_max = abox
        ct = np.round([(x_min + x_max) / 2,
                       (y_min + y_max) / 2]).astype(np.int32)
        h, w = y_max - y_min, x_max - x_min
        abox = np.array(
            [ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2])

        hm = np.zeros([1, snake_config.cp_h, snake_config.cp_w],
                      dtype=np.float32)
        abox_w, abox_h = abox[2] - abox[0], abox[3] - abox[1]
        cp_wh_ = []
        cp_ind_ = []
        ratio = [snake_config.cp_w, snake_config.cp_h] / np.array(
            [abox_w, abox_h])

        decode_boxes = []

        for ex in instance:
            box = np.concatenate([np.min(ex, axis=0), np.max(ex, axis=0)])
            box_w, box_h = box[2] - box[0], box[3] - box[1]
            cp_wh_.append([box_w, box_h])

            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            shift = center - abox[:2]
            ro_center = shift / [abox_w, abox_h
                                 ] * [snake_config.cp_w, snake_config.cp_h]
            ro_center = np.floor(ro_center).astype(np.int32)
            cp_ind_.append(ro_center[1] * hm.shape[2] + ro_center[0])

            ro_box_w, ro_box_h = [box_w, box_h] * ratio
            radius = data_utils.gaussian_radius(
                (math.ceil(ro_box_h), math.ceil(ro_box_w)))
            radius = max(0, int(radius))
            data_utils.draw_umich_gaussian(hm[0], ro_center, radius)

            center = ro_center / [snake_config.cp_w, snake_config.cp_h
                                  ] * [abox_w, abox_h] + abox[:2]
            x_min, y_min = center[0] - box_w / 2, center[1] - box_h / 2
            x_max, y_max = center[0] + box_w / 2, center[1] + box_h / 2
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            decode_boxes.append([x_min, y_min, x_max, y_max])

        cp_hm.append(hm)
        cp_wh.append(cp_wh_)
        cp_ind.append(cp_ind_)

        return decode_boxes

    # def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys,
    #                  c_gt_4pys, h, w):
    #     x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
    #     x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

    #     if np.random.uniform(0, 1) < 0.5:
    #         x_shift = x_min - box[0]
    #         y_shift = y_min - box[1]
    #         box = [
    #             x_min + x_shift, y_min + y_shift, x_max + x_shift,
    #             y_max + y_shift
    #         ]

    #     img_init_poly = dance_cityscapes_utils.get_init(box)
    #     img_init_poly = dance_cityscapes_utils.uniformsample(
    #         img_init_poly, snake_config.init_poly_num)
    #     can_init_poly = dance_cityscapes_utils.img_poly_to_can_poly(
    #         img_init_poly, x_min, y_min, x_max, y_max)
    #     img_gt_poly = extreme_point
    #     can_gt_poly = dance_cityscapes_utils.img_poly_to_can_poly(
    #         img_gt_poly, x_min, y_min, x_max, y_max)

    #     i_it_4pys.append(img_init_poly)
    #     c_it_4pys.append(can_init_poly)
    #     i_gt_4pys.append(img_gt_poly)
    #     c_gt_4pys.append(can_gt_poly)

    def prepare_snake_evolution(self, poly, extreme_point, img_init_polys,
                                can_init_polys, img_gt_polys, can_gt_polys):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = dance_cityscapes_utils.get_octagon(extreme_point)
        img_init_poly = dance_cityscapes_utils.uniformsample(
            octagon, snake_config.poly_num)
        can_init_poly = dance_cityscapes_utils.img_poly_to_can_poly(
            img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = dance_cityscapes_utils.uniformsample(
            poly,
            len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(
            np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = dance_cityscapes_utils.img_poly_to_can_poly(
            img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def prepare_dance_evolution(self, poly, box, img_init_polys, img_gt_polys,
                                whs, img_h, img_w):
        if Polygon(poly).exterior.is_ccw:
            print('poly not in cycle order, discard this poly')
            return
        x_min, y_min, x_max, y_max = box
        # use this as a scaling
        ws = x_max - x_min
        hs = y_max - y_min

        # 1) initial contour
        rectangle = dance_cityscapes_utils.get_rectangle(
            x_min, y_min, x_max, y_max)
        img_init_poly = dance_cityscapes_utils.uniformsample(
            rectangle, snake_config.poly_num)
        # clamp
        img_init_poly[:, 0] = np.clip(img_init_poly[:, 0], 0, img_w - 1)
        img_init_poly[:, 1] = np.clip(img_init_poly[:, 1], 0, img_h - 1)

        # can_init_poly = dance_cityscapes_utils.img_poly_to_can_poly(
        #     img_init_poly, x_min, y_min, x_max, y_max)

        # 2) deformation target
        img_gt_poly = dance_cityscapes_utils.uniformsample(
            poly,
            len(poly) * snake_config.gt_poly_num * dance_config.up_rate)

        tt_idx = np.argmin(
            np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(
            img_gt_poly, -tt_idx,
            axis=0)[::len(poly)]  # still over-sampled by up_rate

        img_gt_poly, aux_ext_idxs = dance_cityscapes_utils.get_aux_extreme_points(
            img_gt_poly)
        tt_idx = np.argmin(
            np.power(img_init_poly - img_gt_poly[0], 2).sum(axis=1))
        img_init_poly = np.roll(img_init_poly, -tt_idx, axis=0)
        img_gt_poly = dance_cityscapes_utils.single_uniform_multisegment_matching(
            img_gt_poly, img_init_poly, aux_ext_idxs, dance_config.up_rate,
            snake_config.poly_num)

        # clamp
        img_gt_poly[:, 0] = np.clip(img_gt_poly[:, 0], 0, img_w - 1)
        img_gt_poly[:, 1] = np.clip(img_gt_poly[:, 1], 0, img_h - 1)

        # can_gt_poly = dance_cityscapes_utils.img_poly_to_can_poly(
        #     img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        # can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        # can_gt_polys.append(can_gt_poly)
        whs.append(np.array([ws, hs]))

    def __getitem__(self, index):
        ann = self.anns[index]

        anno, path, img_id = dance_cityscapes_utils.process_info(
            ann, self.data_root)

        edge_map = cv2.imread(
            path_join(self.edge_map_root,
                      path.split('/')[-1]), cv2.IMREAD_GRAYSCALE)

        img, instance_polys, cls_ids = self.read_original_data(anno, path)

        width = img.shape[1]

        # data augmentation
        orig_img, inp, aug_edge, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            dance_cityscapes_utils.augment(
                img, edge_map, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
        instance_polys = self.transform_original_data(instance_polys, flipped,
                                                      width, trans_output,
                                                      inp_out_hw)

        instance_polys = self.get_valid_polys(instance_polys)
        extreme_points = self.get_extreme_points(instance_polys)
        boxes = self.get_amodal_boxes(extreme_points)

        # image
        img_h, img_w = inp.shape[1:]
        # detection
        output_h, output_w = inp_out_hw[2:]

        act_hm = np.zeros([8, output_h, output_w], dtype=np.float32)
        awh = []
        act_ind = []

        # component
        cp_hm = []
        cp_wh = []
        cp_ind = []

        # # init
        # i_it_4pys = []
        # c_it_4pys = []
        # i_gt_4pys = []
        # c_gt_4pys = []

        # original evolution
        # i_it_pys = []
        # c_it_pys = []
        # i_gt_pys = []
        # c_gt_pys = []

        # dance evolution
        init_box = []
        # init_box_norm = []
        targ_poly = []
        # targ_poly_norm = []
        whs = []  # scale - better normalization

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]
            self.prepare_adet(boxes[i], act_hm, cls_id, awh, act_ind)
            # amodel boxes: [[x_min, y_min, x_max, y_max], ...]
            _ = self.prepare_rcnn(boxes[i], instance_points, cp_hm, cp_wh,
                                  cp_ind)

            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                # extreme_point = instance_points[j]

                x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue

                # self.prepare_init(decode_boxes[j], extreme_point, i_it_4pys,
                #                   c_it_4pys, i_gt_4pys, c_gt_4pys, output_h,
                #                   output_w)
                # self.prepare_snake_evolution(poly, extreme_point, i_it_pys,
                #                              c_it_pys, i_gt_pys, c_gt_pys)
                self.prepare_dance_evolution(
                    poly,
                    bbox,
                    init_box,  # decode_boxes[j] is slightly shifted
                    targ_poly,
                    whs,
                    img_h,
                    img_w)

        # the meaning of the returned data
        # inp: image
        # act_hm: 'ct_hm' means the heatmap of the object center; 'a' means 'amodal', which includes the complete object
        # awh: 'wh' means the width and height of the object bounding box
        # act_ind: the index in an image, row * width + col
        # cp_hm: component heatmap
        # cp_ind: the index in an RoI
        # i_it_4py: initial 4-vertex polygon for extreme point prediction, 'i' means 'image', 'it' means 'initial'
        # c_it_4py: normalized initial 4-vertex polygon. 'c' means 'canonical', which indicates that the polygon coordinates are normalized.
        # i_gt_4py: ground-truth 4-vertex polygon.
        # i_it_py: initial n-vertex polygon for contour deformation.

        ret = {'inp': inp}
        adet = {'act_hm': act_hm, 'awh': awh, 'act_ind': act_ind}
        cp = {'cp_hm': cp_hm, 'cp_wh': cp_wh, 'cp_ind': cp_ind}
        # init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        # snake_evolution = {
        #     'i_it_py': i_it_pys,
        #     'c_it_py': c_it_pys,
        #     'i_gt_py': i_gt_pys,
        #     'c_gt_py': c_gt_pys
        # }
        dance_evolution = {
            'init_box': init_box,
            'targ_poly': targ_poly,
            'whs': whs
        }

        ret.update(adet)
        ret.update(cp)
        # ret.update(init)
        # ret.update(snake_evolution)
        ret.update(dance_evolution)

        # visualization
        # visualize_utils.visualize_snake_detection(orig_img, ret)
        # visualize_utils.visualize_cp_detection(orig_img, ret)
        # visualize_utils.visualize_snake_evolution(orig_img, snake_evolution)
        # visualize_utils.visualize_dance_evolution(orig_img, ret)
        # visualize_utils.visualize_dance_edge(aug_edge)

        act_num = len(act_ind)
        ct_num = len(targ_poly)  # some of i_gt_pys may be discarded
        meta = {
            'center': center,
            'scale': scale,
            'img_id': img_id,
            'ann': ann,
            'act_num': act_num,
            'ct_num': ct_num
        }
        ret.update({'meta': meta})

        ret.update({'edge': aug_edge})
        return ret

    def __len__(self):
        return len(self.anns)
