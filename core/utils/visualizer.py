import numpy as np
from detectron2.utils.visualizer import (
    Visualizer, ColorMode, GenericMask,
    _create_text_labels, _SMALL_OBJECT_AREA_THRESH
)
import pycocotools.mask as mask_util
from detectron2.utils.colormap import random_color

from core.structures.pointset import ExtremePoints


def get_polygon_rles(polygons, image_shape):
    # input: N x (p*2)
    polygons = polygons.cpu().numpy()
    h, w = image_shape
    rles = [
        mask_util.merge(mask_util.frPyObjects([p.tolist()], h, w))
        for p in polygons
    ]
    return rles


class ExVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale=scale, instance_mode=instance_mode)

    def draw_instance_predictions(self, predictions):
        """
        :param predictions:
        :return: Besides the functions of its mother class method, this method deals with extreme points.
        """
        ext_points = predictions.ext_points if predictions.has("ext_points") else None
        pred_polys = predictions.pred_polys if predictions.has("pred_polys") else None
        if False:
            return super().draw_instance_predictions(predictions)
        else:
            boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes if predictions.has("pred_classes") else None
            labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
            keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

            if predictions.has("pred_masks"):
                masks = np.asarray(predictions.pred_masks)
                masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
            else:
                if predictions.has("pred_polys"):
                    output_height = predictions.image_size[0]
                    output_width = predictions.image_size[1]
                    pred_masks = get_polygon_rles(predictions.pred_polys.flatten(),
                                                  (output_height, output_width))

                    masks = np.asarray(pred_masks)
                    masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
                else:
                    masks = None

            path = predictions.pred_path.numpy() if predictions.has("pred_path") else None

            if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
                colors = [
                    self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
                ]
                alpha = 0.8
            else:
                colors = None
                alpha = 0.5

            if self._instance_mode == ColorMode.IMAGE_BW:
                assert predictions.has("pred_masks"), "ColorMode.IMAGE_BW requires segmentations"
                self.output.img = self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                )
                alpha = 0.3

            self.overlay_instances(
                masks=masks,
                boxes=boxes,
                labels=labels,
                ext_points=ext_points,
                path=path,
                keypoints=keypoints,
                assigned_colors=colors,
                alpha=alpha,
            )
            return self.output

    def draw_extreme_pts(self, pts_coord, circle_color, radius=2):
        for pt in pts_coord:
            x, y = pt
            self.draw_circle([x, y], color=circle_color, radius=radius)
        return self.output

    def draw_snake_path(self, path, color, alpha=0.7):
        # path (4, num_points, 2)
        for i, poly in enumerate(path):
            if i > 0:
                prev_poly = path[i - 1]
                offsets = poly - prev_poly
                for j in range(len(offsets)):
                    self.output.ax.arrow(prev_poly[j, 0],
                                         prev_poly[j, 1],
                                         offsets[j, 0],
                                         offsets[j, 1],
                                         linestyle='-',
                                         linewidth=1,
                                         alpha=alpha)
            self.output.ax.plot(poly[0:, 0],
                                poly[0:, 1],
                                color=color,
                                marker='1',
                                alpha=alpha)
        return self.output

    def _convert_ext_points(self, ext_points):
        if isinstance(ext_points, ExtremePoints):
            return ext_points.tensor.numpy()
        else:
            return np.asarray(ext_points)

    def overlay_instances(
            self,
            *,
            boxes=None,
            labels=None,
            masks=None,
            ext_points=None,
            path=None,
            keypoints=None,
            assigned_colors=None,
            alpha=0.5
    ):
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if ext_points is not None:
            ext_points = self._convert_ext_points(ext_points)
            if num_instances:
                assert len(ext_points) == num_instances
            else:
                num_instances = len(ext_points)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if ext_points is not None:
                self.draw_extreme_pts(ext_points[i], circle_color=color, radius=3)

            if path is not None:
                self.draw_snake_path(path[i], color=color)

            if labels is not None:
                # first get a box
                # boxes = None
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion

                text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                horiz_align = "center"

                instance_area = (y1 - y0) * (x1 - x0)
                if (
                        instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                        or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.5
                        * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output
