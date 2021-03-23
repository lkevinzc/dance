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