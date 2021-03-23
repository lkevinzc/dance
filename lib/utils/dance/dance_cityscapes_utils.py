import glob
import os
import json
import numpy as np
from lib.utils import data_utils
import cv2
from lib.utils.snake import snake_config
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize
from scipy import ndimage
import torch
import math

# import logging
# logger = logging.getLogger("DataUtils")

# Globals ----------------------------------------------------------------------
COCO_LABELS = {24: 1, 26: 2, 27: 3, 25: 4, 33: 5, 32: 6, 28: 7, 31: 8}

# Label number to name and color
INSTANCE_LABELS = {
    26: {
        'name': 'car',
        'color': [0, 0, 142]
    },
    24: {
        'name': 'person',
        'color': [220, 20, 60]
    },
    25: {
        'name': 'rider',
        'color': [255, 0, 0]
    },
    32: {
        'name': 'motorcycle',
        'color': [0, 0, 230]
    },
    33: {
        'name': 'bicycle',
        'color': [119, 11, 32]
    },
    27: {
        'name': 'truck',
        'color': [0, 0, 70]
    },
    28: {
        'name': 'bus',
        'color': [0, 60, 100]
    },
    31: {
        'name': 'train',
        'color': [0, 80, 100]
    }
}

# Label name to number
LABEL_DICT = {
    'car': 26,
    'person': 24,
    'rider': 25,
    'motorcycle': 32,
    'bicycle': 33,
    'truck': 27,
    'bus': 28,
    'train': 31
}
# LABEL_DICT = {'bicycle': 33}

# Label name to contiguous number
JSON_DICT = dict(car=0,
                 person=1,
                 rider=2,
                 motorcycle=3,
                 bicycle=4,
                 truck=5,
                 bus=6,
                 train=7)
# JSON_DICT = dict(bicycle=0)
# Contiguous number to name
NUMBER_DICT = {
    0: 'car',
    1: 'person',
    2: 'rider',
    3: 'motorcycle',
    4: 'bicycle',
    5: 'truck',
    6: 'bus',
    7: 'train'
}
# NUMBER_DICT = {0:'bicycle'}
# Array of keys
KEYS = np.array([[26000, 26999], [24000, 24999], [25000,
                                                  25999], [32000, 32999],
                 [33000, 33999], [27000, 27999], [28000, 28999],
                 [31000, 31999]])

NUM_CLASS = {
    'person': 17914,
    'rider': 1755,
    'car': 26944,
    'truck': 482,
    'bus': 379,
    'train': 168,
    'motorcycle': 735,
    'bicycle': 3658
}

# ------------------------------------------------------------------------------


def continuous_label_to_cityscapes_label(labels):
    return np.array([LABEL_DICT[NUMBER_DICT[label]] for label in labels])


def read_dataset(ann_files):
    if not isinstance(ann_files, tuple):
        ann_files = (ann_files, )

    ann_file = []
    for ann_file_dir in ann_files:
        ann_file += glob.glob(os.path.join(ann_file_dir, '*/*.json'))

    ann_filter = []
    for fname in ann_file:
        with open(fname, 'r') as f:
            ann = json.load(f)
            examples = []
            for instance in ann:
                instance_label = instance['label']
                if instance_label not in LABEL_DICT:
                    continue
                examples.append(instance)
            if len(examples) > 0:
                ann_filter.append(fname)
    return ann_filter


def process_info(fname, data_root):
    with open(fname, 'r') as f:
        ann = json.load(f)

    examples = []

    for instance in ann:
        instance_label = instance['label']
        if instance_label not in LABEL_DICT:
            continue
        examples.append(instance)

    img_path = os.path.join(data_root,
                            '/'.join(ann[0]['img_path'].split('/')[-3:]))
    img_id = ann[0]['image_id']

    return examples, img_path, img_id


def xywh_to_xyxy(boxes):
    """
    boxes: [[x, y, w, h]]
    """
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return boxes
    x, y, w, h = np.split(boxes, 4, axis=1)
    x_max = x + w
    y_max = y + h
    return np.concatenate([x, y, x_max, y_max], axis=1)


def augment(img, edge_map, split, _data_rng, _eig_val, _eig_vec, mean, std,
            polys):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = snake_config.scale
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    # random crop and flip augmentation
    flipped = False
    if split == 'train':
        scale = scale * np.random.uniform(0.4, 1.6)
        seed = np.random.randint(0, len(polys))
        index = np.random.randint(0, len(polys[seed]))
        poly = polys[seed][index]['poly']
        center[0], center[1] = poly[np.random.randint(len(poly))]
        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2
        center[0] = np.clip(center[0], a_min=border, a_max=width - border)
        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        center[1] = np.clip(center[1], a_min=border, a_max=height - border)

        # flip augmentation
        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            edge_map = edge_map[:, ::-1]
            center[0] = width - center[0] - 1

    input_w, input_h = snake_config.input_w, snake_config.input_h
    if split != 'train':
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        # input_w, input_h = width, height
        input_w, input_h = int((width / 0.85 + 31) // 32 * 32), int(
            (height / 0.85 + 31) // 32 * 32)

    trans_input = data_utils.get_affine_transform(center, scale, 0,
                                                  [input_w, input_h])
    inp = cv2.warpAffine(img,
                         trans_input, (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    aug_edge = cv2.warpAffine(edge_map,
                              trans_input, (input_w, input_h),
                              flags=cv2.INTER_LINEAR,
                              borderValue=255)

    # color augmentation
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)
        # data_utils.blur_aug(inp)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // snake_config.down_ratio, input_w // snake_config.down_ratio
    trans_output = data_utils.get_affine_transform(center, scale, 0,
                                                   [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    return orig_img, inp, aug_edge, trans_input, trans_output, flipped, center, scale, inp_out_hw


def transform_bbox(bbox, trans_output, output_h, output_w):
    bbox = data_utils.affine_transform(bbox.reshape(-1, 2),
                                       trans_output).ravel()
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
    return bbox


def handle_break_point(poly, axis, number, outside_border):
    if len(poly) == 0:
        return []

    if len(poly[outside_border(poly[:, axis], number)]) == len(poly):
        return []

    break_points = np.argwhere(
        outside_border(poly[:-1, axis], number) != outside_border(
            poly[1:, axis], number)).ravel()
    if len(break_points) == 0:
        return poly

    new_poly = []
    if not outside_border(poly[break_points[0], axis], number):
        new_poly.append(poly[:break_points[0]])

    for i in range(len(break_points)):
        current_poly = poly[break_points[i]]
        next_poly = poly[break_points[i] + 1]
        mid_poly = current_poly + (next_poly - current_poly) * (
            number - current_poly[axis]) / (next_poly[axis] -
                                            current_poly[axis])

        if outside_border(poly[break_points[i], axis], number):
            if mid_poly[axis] != next_poly[axis]:
                new_poly.append([mid_poly])
            next_point = len(poly) if i == (len(break_points) -
                                            1) else break_points[i + 1]
            new_poly.append(poly[break_points[i] + 1:next_point])
        else:
            new_poly.append([poly[break_points[i]]])
            if mid_poly[axis] != current_poly[axis]:
                new_poly.append([mid_poly])

    if outside_border(poly[-1, axis], number) != outside_border(
            poly[0, axis], number):
        current_poly = poly[-1]
        next_poly = poly[0]
        mid_poly = current_poly + (next_poly - current_poly) * (
            number - current_poly[axis]) / (next_poly[axis] -
                                            current_poly[axis])
        new_poly.append([mid_poly])

    return np.concatenate(new_poly)


def transform_polys(polys, trans_output, output_h, output_w):
    new_polys = []
    for i in range(len(polys)):
        poly = polys[i]
        poly = data_utils.affine_transform(poly, trans_output)
        poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
        poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
        if len(poly) == 0:
            continue
        if len(np.unique(poly, axis=0)) <= 2:
            continue
        new_polys.append(poly)
    return new_polys


def get_valid_shape_poly(poly):
    """a poly may be self-intersected"""
    shape_poly = Polygon(poly)
    if shape_poly.is_valid:
        if shape_poly.area < 5:
            return []
        else:
            return [shape_poly]

    # self-intersected situation
    linering = shape_poly.exterior

    # disassemble polygons from multiple line strings
    mls = linering.intersection(linering)
    # assemble polygons from multiple line strings
    polygons = polygonize(mls)
    multi_shape_poly = MultiPolygon(polygons)
    shape_polys = []
    for shape_poly in multi_shape_poly:
        if shape_poly.area < 5:
            continue
        shape_polys.append(shape_poly)
    return shape_polys


def get_valid_polys(polys):
    """create shape_polys and filter polys"""
    # convert polygons into shape_poly
    shape_polys = []
    for poly in polys:
        shape_polys.extend(get_valid_shape_poly(poly))

    # remove polys being contained
    n = len(shape_polys)
    relation = np.zeros([n, n], dtype=np.bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            relation[i, j] = shape_polys[j].contains(shape_polys[i])

    relation = np.any(relation, axis=1)
    shape_polys = [
        shape_polys[i] for i, shape_poly in enumerate(shape_polys)
        if not relation[i]
    ]
    polys = [
        np.array(shape_poly.exterior.coords)[::-1]
        if shape_poly.exterior.is_ccw else np.array(shape_poly.exterior.coords)
        for shape_poly in shape_polys
    ]
    return polys


def filter_tiny_polys(polys):
    polys_ = []
    for poly in polys:
        x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
        x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
        if x_max - x_min >= 1 and y_max - y_min >= 1:
            polys_.append(poly)
    return [poly for poly in polys_ if Polygon(poly).area > 5]


def get_cw_polys(polys):
    return [
        poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys
    ]


def polygon_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(int)], 1)
    return mask


def get_inner_center(mask, mask_ct, h_int, w_int):
    mask_ct_int = np.round(mask_ct).astype(np.int32)
    if snake_config.box_center and mask[mask_ct_int[1], mask_ct_int[0]] == 1:
        ct = mask_ct_int
    else:
        dt = ndimage.distance_transform_edt(mask)
        dt_max = ndimage.maximum_filter(dt, footprint=np.ones([3, 3]))
        mask = (dt_max == dt) * mask

        radius = np.sqrt(h_int * h_int + w_int * w_int) / 6
        xy = np.argwhere(mask)[:, [1, 0]]
        dt = dt[xy[:, 1], xy[:, 0]]
        ct_distance = np.sqrt(np.power(xy - mask_ct, 2).sum(1))
        inlier = ct_distance < radius
        if snake_config.center_scope and len(np.argwhere(inlier)) > 0:
            xy = xy[inlier]
            dt = dt[inlier]
            xy = xy[np.argwhere(dt == np.max(dt)).ravel()]
            ct = xy[np.power(xy - mask_ct, 2).sum(1).argmin()]
        else:
            xy = np.argwhere(mask)[:, [1, 0]]
            ct = xy[np.power(xy - mask_ct, 2).sum(1).argmin()]
    return mask_ct_int


def prepare_ct_off_mask(mask_poly, mask_ct, h_int, w_int):
    mask_poly = mask_poly + 1
    mask_ct = mask_ct + 1
    mask = polygon_to_mask(mask_poly, h_int + 2, w_int + 2)
    ct = get_inner_center(mask, mask_ct, h_int, w_int) - 1
    mask = mask[1:-1, 1:-1]
    xy = np.argwhere(mask)[:, [1, 0]]
    off = ct - xy
    return ct, off, xy


def uniform_sample_1d(pts, new_n):
    if new_n == 1:
        return pts[:1]
    n = pts.shape[0]
    if n == new_n + 1:
        return pts[:-1]
    # len: n - 1
    segment_len = np.sqrt(np.sum((pts[1:] - pts[:-1])**2, axis=1))

    # down-sample or up-sample
    # n
    start_node = np.cumsum(np.concatenate([np.array([0]), segment_len]))
    total_len = np.sum(segment_len)

    new_per_len = total_len / new_n

    mark_1d = ((np.arange(new_n - 1) + 1) * new_per_len).reshape(-1, 1)
    locate = (start_node.reshape(1, -1) - mark_1d)
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


def reorder_perloss(oct_sampled_targets, oct_sampled_pts):
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
    enumerated_ind = torch.fmod(ind2 + ind1.view(-1, 1),
                                num_locs).view(-1).long()
    enumerated_targets = oct_sampled_targets[:, enumerated_ind, :].view(
        n, -1, num_locs, 2)
    diffs = enumerated_targets - oct_sampled_pts[:, None, ...]
    diffs_sum = diffs.pow(2).sum(3).sum(2)
    tt_idx = torch.argmin(diffs_sum, dim=1)
    re_ordered_gt = enumerated_targets[torch.arange(n), tt_idx]
    return re_ordered_gt


def single_uniform_multisegment_matching(dense_targets, sampled_pts, ext_idx,
                                         up_rate, poly_num):
    """
    Several points to note while debugging:
    1) For GT (from which points are sampled), include both end by [s, e + 1] indexing.
    2) If GT not increasing (equal), shift forwards by 1.
    3) Check the 1st sampled point is indexed by 0.
    4) Check the last sampled point is NOT indexed by 0 or any small value.
    """
    min_idx = ext_idx

    ch_pts = dense_targets[min_idx]  # characteristic points

    diffs = ((ch_pts[:, np.newaxis, :] -
              sampled_pts[np.newaxis])**2).sum(axis=2)
    ext_idx = np.argmin(diffs, axis=1)
    if ext_idx[0] != 0:
        ext_idx[0] = 0
    if ext_idx[-1] < ext_idx[1]:
        ext_idx[-1] = poly_num - 2
    ext_idx = np.sort(ext_idx)

    aug_ext_idx = np.concatenate([ext_idx, np.array([poly_num])], axis=0)

    # diff = np.sum((ch_pts[:, np.newaxis, :] - dense_targets[np.newaxis, :, :]) ** 2, axis=2)
    # min_idx = np.argmin(diff, axis=1)

    aug_min_idx = np.concatenate(
        [min_idx, np.array([poly_num * up_rate])], axis=0)

    if aug_min_idx[-1] < aug_min_idx[1]:
        print("WARNING: Last point not matching!")
        print(aug_ext_idx)
        print(aug_min_idx)
        aug_min_idx[
            -1] = poly_num * up_rate - 2  # enforce matching of the last point

    if aug_min_idx[0] != 0:
        # TODO: This is crucial, or other wise the first point may be
        # TODO: matched to near 640, then sorting will completely mess
        print("WARNING: First point not matching!")
        print(aug_ext_idx)
        print(aug_min_idx)
        aug_min_idx[0] = 0  # enforce matching of the 1st point

    aug_ext_idx = np.sort(aug_ext_idx)
    aug_min_idx = np.sort(aug_min_idx)

    # === error-prone ===

    # deal with corner cases

    if aug_min_idx[-2] == poly_num * up_rate - 1:
        # print("WARNING: Bottom extreme point being the last point!")
        print("WARNING: Bottom extreme point being the last point!")
        # hand designed remedy
        aug_min_idx[-2] = poly_num * up_rate - 3
        aug_min_idx[-1] = poly_num * up_rate - 2

    if aug_min_idx[-1] == poly_num * up_rate - 1:
        # print("WARNING: Right extreme point being the last point!")
        print("WARNING: Right extreme point being the last point!")
        print(aug_ext_idx)
        print(aug_min_idx)
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
                    [dense_targets[aug_min_idx[i]:], dense_targets[:1]],
                    axis=0)
            else:
                pts = dense_targets[aug_min_idx[i]:aug_min_idx[i + 1] +
                                    1]  # including
            new_sampled_pts = uniform_sample_1d(
                pts, aug_ext_idx[i + 1] - aug_ext_idx[i])
            segments.append(new_sampled_pts)
        # segments.append(dense_targets[-1:]) # close the loop
        segments = np.concatenate(segments, axis=0)
        if len(segments) != poly_num:
            # print("WARNING: Number of points not matching!")
            print("WARNING: Number of points not matching!",
                        len(segments))
            raise ValueError(len(segments))
    except Exception as err:  # may exist some very tricky corner cases...
        # print("WARNING: Tricky corner cases occurred!")
        print("WARNING: Tricky corner cases occurred!")
        print(err)
        print(aug_ext_idx)
        print(aug_min_idx)
        # raise ValueError('TAT')
        segments = reorder_perloss(
            torch.from_numpy(dense_targets[::up_rate][None]),
            torch.from_numpy(sampled_pts)[None])[0]
        segments = segments.numpy()

    return segments


def get_aux_extreme_points(pts):
    num_pt = pts.shape[0]

    aux_ext_pts = []

    l, t = min(pts[:, 0]), min(pts[:, 1])
    r, b = max(pts[:, 0]), max(pts[:, 1])
    # 3 degrees
    thresh = 0.02
    band_thresh = 0.02
    w = r - l + 1
    h = b - t + 1

    t_band = np.where((pts[:, 1] - t) <= band_thresh * h)[0].tolist()
    while t_band:
        t_idx = t_band[np.argmin(pts[t_band, 1])]
        t_idxs = [t_idx]
        tmp = (t_idx + 1) % num_pt
        while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
            t_idxs.append(tmp)
            tmp = (tmp + 1) % num_pt
        tmp = (t_idx - 1) % num_pt
        while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
            t_idxs.append(tmp)
            tmp = (tmp - 1) % num_pt
        tt = (max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) / 2
        aux_ext_pts.append(np.array([tt, t]))
        t_band = [item for item in t_band if item not in t_idxs]

    b_band = np.where((b - pts[:, 1]) <= band_thresh * h)[0].tolist()
    while b_band:
        b_idx = b_band[np.argmax(pts[b_band, 1])]
        b_idxs = [b_idx]
        tmp = (b_idx + 1) % num_pt
        while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
            b_idxs.append(tmp)
            tmp = (tmp + 1) % num_pt
        tmp = (b_idx - 1) % num_pt
        while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
            b_idxs.append(tmp)
            tmp = (tmp - 1) % num_pt
        bb = (max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) / 2
        aux_ext_pts.append(np.array([bb, b]))
        b_band = [item for item in b_band if item not in b_idxs]

    l_band = np.where((pts[:, 0] - l) <= band_thresh * w)[0].tolist()
    while l_band:
        l_idx = l_band[np.argmin(pts[l_band, 0])]
        l_idxs = [l_idx]
        tmp = (l_idx + 1) % num_pt
        while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
            l_idxs.append(tmp)
            tmp = (tmp + 1) % num_pt
        tmp = (l_idx - 1) % num_pt
        while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
            l_idxs.append(tmp)
            tmp = (tmp - 1) % num_pt
        ll = (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) / 2
        aux_ext_pts.append(np.array([l, ll]))
        l_band = [item for item in l_band if item not in l_idxs]

    r_band = np.where((r - pts[:, 0]) <= band_thresh * w)[0].tolist()
    while r_band:
        r_idx = r_band[np.argmax(pts[r_band, 0])]
        r_idxs = [r_idx]
        tmp = (r_idx + 1) % num_pt
        while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
            r_idxs.append(tmp)
            tmp = (tmp + 1) % num_pt
        tmp = (r_idx - 1) % num_pt
        while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
            r_idxs.append(tmp)
            tmp = (tmp - 1) % num_pt
        rr = (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) / 2
        aux_ext_pts.append(np.array([r, rr]))
        r_band = [item for item in r_band if item not in r_idxs]

    # assert len(aux_ext_pts) >= 4
    pt0 = aux_ext_pts[0]

    # collecting
    aux_ext_pts = np.stack(aux_ext_pts, axis=0)

    # ordering
    shift_idx = np.argmin(np.power(pts - pt0, 2).sum(axis=1))
    re_ordered_pts = np.roll(pts, -shift_idx, axis=0)

    # indexing
    ext_idxs = np.argmin(np.sum(
        (aux_ext_pts[:, np.newaxis, :] - re_ordered_pts[np.newaxis, ...])**2,
        axis=2),
                         axis=1)
    ext_idxs[0] = 0

    ext_idxs = np.sort(np.unique(ext_idxs))

    return re_ordered_pts, ext_idxs


def get_extreme_points(pts):
    l, t = min(pts[:, 0]), min(pts[:, 1])
    r, b = max(pts[:, 0]), max(pts[:, 1])
    # 3 degrees
    thresh = 0.02
    w = r - l + 1
    h = b - t + 1

    t_idx = np.argmin(pts[:, 1])
    t_idxs = [t_idx]
    tmp = (t_idx + 1) % pts.shape[0]
    while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
        t_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (t_idx - 1) % pts.shape[0]
    while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
        t_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    tt = [(max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) / 2, t]

    b_idx = np.argmax(pts[:, 1])
    b_idxs = [b_idx]
    tmp = (b_idx + 1) % pts.shape[0]
    while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
        b_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (b_idx - 1) % pts.shape[0]
    while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
        b_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    bb = [(max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) / 2, b]

    l_idx = np.argmin(pts[:, 0])
    l_idxs = [l_idx]
    tmp = (l_idx + 1) % pts.shape[0]
    while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
        l_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (l_idx - 1) % pts.shape[0]
    while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
        l_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    ll = [l, (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) / 2]

    r_idx = np.argmax(pts[:, 0])
    r_idxs = [r_idx]
    tmp = (r_idx + 1) % pts.shape[0]
    while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
        r_idxs.append(tmp)
        tmp = (tmp + 1) % pts.shape[0]
    tmp = (r_idx - 1) % pts.shape[0]
    while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
        r_idxs.append(tmp)
        tmp = (tmp - 1) % pts.shape[0]
    rr = [r, (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) / 2]

    return np.array([tt, ll, bb, rr])


def get_quadrangle(box):
    x_min, y_min, x_max, y_max = box
    quadrangle = [[(x_min + x_max) / 2., y_min], [x_min, (y_min + y_max) / 2.],
                  [(x_min + x_max) / 2., y_max], [x_max, (y_min + y_max) / 2.]]
    return np.array(quadrangle)


def get_box(box):
    x_min, y_min, x_max, y_max = box
    box = [[(x_min + x_max) / 2., y_min], [x_min, y_min],
           [x_min, (y_min + y_max) / 2.], [x_min, y_max],
           [(x_min + x_max) / 2., y_max], [x_max, y_max],
           [x_max, (y_min + y_max) / 2.], [x_max, y_min]]
    return np.array(box)


def get_init(box):
    if snake_config.init == 'quadrangle':
        return get_quadrangle(box)
    else:
        return get_box(box)


def get_octagon(ex):
    w, h = ex[3][0] - ex[1][0], ex[2][1] - ex[0][1]
    t, l, b, r = ex[0][1], ex[1][0], ex[2][1], ex[3][0]
    x = 8.0
    octagon = [
        ex[0][0],
        ex[0][1],
        max(ex[0][0] - w / x, l),
        ex[0][1],
        ex[1][0],
        max(ex[1][1] - h / x, t),
        ex[1][0],
        ex[1][1],
        ex[1][0],
        min(ex[1][1] + h / x, b),
        max(ex[2][0] - w / x, l),
        ex[2][1],
        ex[2][0],
        ex[2][1],
        min(ex[2][0] + w / x, r),
        ex[2][1],
        ex[3][0],
        min(ex[3][1] + h / x, b),
        ex[3][0],
        ex[3][1],
        ex[3][0],
        max(ex[3][1] - h / x, t),
        min(ex[0][0] + w / x, r),
        ex[0][1],
    ]
    return np.array(octagon).reshape(-1, 2)


def get_rectangle(x_min, y_min, x_max, y_max):
    rectangle = [
        x_min,
        y_min,  # top-left
        x_min,
        y_max,  # bottom-left
        x_max,
        y_max,  # bottom-right
        x_max,
        y_min,  # top-right
    ]
    return np.array(rectangle).reshape(-1, 2)


def get_ellipse(box, cp_num):
    pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
    for i in range(cp_num):
        theta = 1.0 * i / cp_num * 2 * np.pi
        x = np.cos(theta)
        y = -np.sin(theta)
        pointsnp[i, 0] = x
        pointsnp[i, 1] = y
    pointsnp /= 2.
    pointsnp += 0.5
    w, h = box[2] - box[0], box[3] - box[1]
    pointsnp *= np.array([w, h])
    pointsnp = pointsnp + np.array([box[0], box[1]])
    return pointsnp


def uniform_sample_init(poly):
    polys = []
    ind = np.array(list(range(0, len(poly), len(poly) // 4)))
    next_ind = np.roll(ind, shift=-1)
    for i in range(len(ind)):
        poly_ = poly[ind[i]:ind[i] + len(poly) // 4]
        poly_ = np.append(poly_, [poly[next_ind[i]]], axis=0)
        poly_ = uniform_sample_segment(poly_, snake_config.init_poly_num // 4)
        polys.append(poly_)
    return np.concatenate(polys)


def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2)**2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(
            np.int32)
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
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(
                -1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp


def uniform_sample_segment(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum - 1, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    pgtnp_px2 = pgtnp_px2[:-1]
    pnum = pnum - 1
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2)**2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(
        np.int32)
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
        pb_1x2 = pgtnp_px2[i:i + 1]
        pe_1x2 = pgtnext_px2[i:i + 1]

        pnewnum = edgenum[i]
        wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(
            -1, 1) / edgenum[i]

        pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
        psample.append(pmids)

    psamplenp = np.concatenate(psample, axis=0)
    return psamplenp


def img_poly_to_can_poly(img_poly, x_min, y_min, x_max, y_max):
    x_min, y_min = np.min(img_poly, axis=0)
    can_poly = img_poly - np.array([x_min, y_min])
    # h, w = y_max - y_min + 1, x_max - x_min + 1
    # long_side = max(h, w)
    # can_poly = can_poly / long_side
    return can_poly


def add_gaussian_noise(poly, x_min, y_min, x_max, y_max):
    h, w = y_max - y_min, x_max - x_min
    radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius))
    noise = np.random.uniform(-radius, radius, poly.shape)
    return poly + noise


def clip_poly_to_image(poly, h, w):
    poly[:, 0] = np.clip(poly[:, 0], a_min=0, a_max=w - 1)
    poly[:, 1] = np.clip(poly[:, 1], a_min=0, a_max=h - 1)
    return poly
