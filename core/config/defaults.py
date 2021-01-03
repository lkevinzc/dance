from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# DANCE (ZC)
# ---------------------------------------------------------------------------- #
_C.MODEL.DANCE = CN()

# utils
_C.MODEL.DANCE.VIS_PATH = False
_C.MODEL.DANCE.MASK_IN = "OCT_RLE"

# ---------------------------------------------------------------------------- #
# DANCE Head (ZC)
# ---------------------------------------------------------------------------- #
_C.MODEL.DANCE.HEAD = CN()

_C.MODEL.DANCE.HEAD.LOSS_WEIGHT = 10

_C.MODEL.DANCE.HEAD.FEAT_DIM = 128

_C.MODEL.DANCE.HEAD.NUM_ITER = (0, 0, 1)  # correspond to the convs
_C.MODEL.DANCE.HEAD.NUM_CONVS = 2

_C.MODEL.DANCE.HEAD.NUM_LAYER = (8, 8, 8)
_C.MODEL.DANCE.HEAD.CIR_DILATIONS = (
    (1, 1, 1, 2, 2, 4, 4),
    (1, 1, 1, 2, 2, 4, 4),
    (1, 1, 1, 2, 2, 4, 4),
)  # by default the first one is 1.

_C.MODEL.DANCE.HEAD.UP_SAMPLE_RATE = 2

_C.MODEL.DANCE.HEAD.LOSS_L1_BETA = 0.11
_C.MODEL.DANCE.HEAD.DILATIONS = (1, 1)

_C.MODEL.DANCE.HEAD.NUM_SAMPLING = 128


# ---------------------------------------------------------------------------- #
# Edge Prediction Head (ZC)
# ---------------------------------------------------------------------------- #
_C.MODEL.DANCE.EDGE = CN()
_C.MODEL.DANCE.EDGE.NAME = "EdgeFPNHead"

_C.MODEL.DANCE.EDGE.TRAIN = True

_C.MODEL.DANCE.EDGE.IN_FEATURES = ["p2"]
_C.MODEL.DANCE.EDGE.STRONG_FEAT = False
# Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
# the correposnding pixel.
_C.MODEL.DANCE.EDGE.IGNORE_VALUE = 255
# Number of classes in the edge prediction head
_C.MODEL.DANCE.EDGE.NUM_CLASSES = 1  # (only foreground or not)
# Number of channels in the 3x3 convs inside semantic-FPN heads.
_C.MODEL.DANCE.EDGE.CONVS_DIM = 256
# Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
_C.MODEL.DANCE.EDGE.COMMON_STRIDE = 4
# Normalization method for the convolution layers. Options: "" (no norm), "GN".
_C.MODEL.DANCE.EDGE.NORM = "GN"
_C.MODEL.DANCE.EDGE.BCE_WEIGHT = (
    0  # 1:1 BCE harms the training, very small BCE not helpful
)

_C.MODEL.DANCE.EDGE.LOSS_WEIGHT = 1

# ---------------------------------------------------------------------------- #
# Investigation Configs (ZC)
# ---------------------------------------------------------------------------- #
_C.TEST.GT_IN = CN()
_C.TEST.GT_IN.ON = False
_C.TEST.GT_IN.WHAT = ["edge", "instance"]  # {"edge", "instance"}

# ---------------------------------------------------------------------------- #
# VoV Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()

_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"

_C.MODEL.VOVNET.OUT_CHANNELS = 256

_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = "giou"
_C.MODEL.FCOS.EXT_LOSS_TYPE = "smoothl1"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.SEED = 77
