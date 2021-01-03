import numpy as np
import torch
from core.structures.pointset import ExtremePoints
from detectron2.structures.boxes import Boxes


# unused
def get_octagon(ex):
    ex = np.array(ex).reshape(4, 2)
    w, h = ex[3][0] - ex[1][0], ex[2][1] - ex[0][1]
    t, l, b, r = ex[0][1], ex[1][0], ex[2][1], ex[3][0]
    x = 8.
    octagon = [[min(ex[0][0] + w / x, r), ex[0][1], \
                max(ex[0][0] - w / x, l), ex[0][1], \
                ex[1][0], max(ex[1][1] - h / x, t), \
                ex[1][0], min(ex[1][1] + h / x, b), \
                max(ex[2][0] - w / x, l), ex[2][1], \
                min(ex[2][0] + w / x, r), ex[2][1], \
                ex[3][0], min(ex[3][1] + h / x, b), \
                ex[3][0], max(ex[3][1] - h / x, t)
                ]]
    return octagon


def get_extreme_points(pts):
    num_pt = pts.shape[0]
    l, t = min(pts[:, 0]), min(pts[:, 1])
    r, b = max(pts[:, 0]), max(pts[:, 1])
    # 3 degrees
    thresh = 0.02
    w = r - l + 1
    h = b - t + 1

    t_idx = np.argmin(pts[:, 1])
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

    b_idx = np.argmax(pts[:, 1])
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

    l_idx = np.argmin(pts[:, 0])
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

    r_idx = np.argmax(pts[:, 0])
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

    return np.array([tt, ll, bb, rr])


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
        (aux_ext_pts[:, np.newaxis, :] - re_ordered_pts[np.newaxis, ...]) ** 2, axis=2),
        axis=1)
    ext_idxs[0] = 0

    ext_idxs = np.sort(np.unique(ext_idxs))

    return re_ordered_pts, ext_idxs


def vis_training_targets(cfg, fcose_outputs, image_list, idx=0):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    colors = np.array([[1, 1, 198],
                       [51, 1, 148],
                       [101, 1, 98],
                       [151, 1, 48],
                       [201, 1, 8]]) / 255.

    num_loc_list = [len(loc) for loc in fcose_outputs.locations]
    fcose_outputs.num_loc_list = num_loc_list

    # compute locations to size ranges
    loc_to_size_range = []
    for l, loc_per_level in enumerate(fcose_outputs.locations):
        loc_to_size_range_per_level = loc_per_level.new_tensor(fcose_outputs.sizes_of_interest[l])
        loc_to_size_range.append(
            loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
        )

    # (Sigma_{levels_points}, 2)
    loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
    locations = torch.cat(fcose_outputs.locations, dim=0)

    training_targets = fcose_outputs.compute_targets_for_locations(
        locations, fcose_outputs.gt_instances, loc_to_size_range
    )

    training_target = {k: v[idx] for k, v in training_targets.items()}

    fig, ax = plt.subplots(1, figsize=(20, 10))
    fig.tight_layout()

    labels = training_target['labels']
    reg_targets = training_target['reg_targets']
    ext_targets = training_target['ext_targets']

    idxOfloc_of_interest = torch.where(labels != 20)[0]

    global locxys, reg_targets_oi, ext_targets_oi, detections

    locxys = locations[idxOfloc_of_interest]

    reg_targets_oi = reg_targets[idxOfloc_of_interest]
    ext_targets_oi = ext_targets[idxOfloc_of_interest]

    detections = torch.stack([
        locxys[:, 0] - reg_targets_oi[:, 0],
        locxys[:, 1] - reg_targets_oi[:, 1],
        locxys[:, 0] + reg_targets_oi[:, 2],
        locxys[:, 1] + reg_targets_oi[:, 3],
    ], dim=1)

    global tmp, ext_points

    ext_points = ExtremePoints.from_boxes(Boxes(detections),
                                          ext_targets_oi,
                                          locxys).tensor.cpu().numpy()

    tmp = ext_points

    im = image_list.tensor[idx]
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(im.device).view(-1, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(im.device).view(-1, 1, 1)
    im_norm = ((im * pixel_std) + pixel_mean).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)

    ax.imshow(im_norm)
    locxys_np = locxys.cpu().numpy()
    reg_targets_oi_np = reg_targets_oi.cpu().numpy()
    ext_targets_oi_np = ext_targets_oi.cpu().numpy()
    detections_np = detections.cpu().numpy()

    for i in range(len(locxys_np)):
        ax.scatter(locxys_np[i, 0], locxys_np[i, 1], color=colors[i % len(colors)].tolist(), marker='*')
        x1, y1, x2, y2 = detections_np[i, :]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=colors[i % len(colors)].tolist(),
                                 facecolor='none', fill=False)
        ax.add_patch(rect)

        ax.scatter(ext_points[i][:, 0], ext_points[i][:, 1], color=colors[i % len(colors)].tolist(), marker='+')

    plt.show()
