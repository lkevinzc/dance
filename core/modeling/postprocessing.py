from detectron2.structures import Instances
from core.structures import PolygonPoints


def detector_postprocess(results,
                         output_height,
                         output_width):
    # the results.image_size here is the one the model saw, typically (800, xxxx)
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    # now the results.image_size is the one of raw input image
    output_boxes.clip(results.image_size)
    results = results[output_boxes.nonempty()]

    if results.has("pred_polys"):
        if results.has("pred_path"):
            snake_path = results.pred_path
            for i in range(snake_path.size(1)):  # number of evolution
                current_poly = PolygonPoints(snake_path[:, i, :, :])
                current_poly.scale(scale_x, scale_y)
                current_poly.clip(results.image_size)
                snake_path[:, i, :, :] = current_poly.tensor
        # TODO: Note that we did not scale exts (no need if not for evaluation)
        if results.has("ext_points"):
            results.ext_points.scale(scale_x, scale_y)
        results.pred_polys.scale(scale_x, scale_y)
        return results
    return results


def edge_map_postprocess(result, img_size):
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    return result[0][0]
