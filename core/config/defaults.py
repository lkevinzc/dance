from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.USE_VOVNET = False

# ---------------------------------------------------------------------------- #
# MY CONFIG (ZC)
# ---------------------------------------------------------------------------- #
_C.MODEL.DANCE = CN()

# Channeling the input for mask_pred used for model evaluation
# Use NO to avoid error during evaluation when turn MASK_ON but no mask_pred output.
_C.MODEL.DANCE.MASK_IN = "OCT_RLE"  # {'BOX', 'OCT_BIT', 'OCT_RLE', 'MASK', 'NO'}
_C.MODEL.DANCE.SEMANTIC_FILTER = False
_C.MODEL.DANCE.SEMANTIC_FILTER_TH = 0.1
_C.MODEL.DANCE.ROI_SIZE = 28


_C.MODEL.DANCE.RE_COMP_BOX = False

# ---------------------------------------------------------------------------- #
# Deformable Convolution Head (ZC)
# ---------------------------------------------------------------------------- #
_C.MODEL.DEFORM_HEAD = CN()
_C.MODEL.DEFORM_HEAD.ON = False
_C.MODEL.DEFORM_HEAD.NUM_CONVS = 256
_C.MODEL.DEFORM_HEAD.NORM = "GN"
_C.MODEL.DEFORM_HEAD.USE_MODULATED = False

# ---------------------------------------------------------------------------- #
# Snake Head (ZC)
# ---------------------------------------------------------------------------- #
_C.MODEL.SNAKE_HEAD = CN()

_C.MODEL.SNAKE_HEAD.DETACH = False


_C.MODEL.SNAKE_HEAD.ORIGINAL = False

_C.MODEL.SNAKE_HEAD.STRUCTURE = "sequential"  # {"sequential", "parallel"};

# circular conv net / graph conv net
_C.MODEL.SNAKE_HEAD.CONV_TYPE = "ccn"  # {"ccn", "gcn"};

_C.MODEL.SNAKE_HEAD.FEAT_DIM = 128

_C.MODEL.SNAKE_HEAD.NUM_ITER = (0, 0, 1)  # correspond to the convs
_C.MODEL.SNAKE_HEAD.NUM_CONVS = 2
_C.MODEL.SNAKE_HEAD.STRONGER = False

_C.MODEL.SNAKE_HEAD.MULTI_OFFSET = 1

_C.MODEL.SNAKE_HEAD.SKIP = False
_C.MODEL.SNAKE_HEAD.NUM_LAYER = (8, 8, 8)
_C.MODEL.SNAKE_HEAD.CIR_DILATIONS = (
    (1, 1, 1, 2, 2, 4, 4),
    (1, 1, 1, 2, 2, 4, 4),
    (1, 1, 1, 2, 2, 4, 4),
)  # by default the first one is 1.

_C.MODEL.SNAKE_HEAD.MSCORE_SNAKE_ON = False
_C.MODEL.SNAKE_HEAD.MSCORE_SNAKE_NUM_LAYER = 5
_C.MODEL.SNAKE_HEAD.MSCORE_SNAKE_CIR_DILATIONS = (
    1,
    2,
    2,
    4,
)  # by default the first one is 1.
_C.MODEL.SNAKE_HEAD.MSCORE_SNAKE_FEAT_DIM = 128
_C.MODEL.SNAKE_HEAD.MSCORE_SNAKE_MIN_AREA = 5 * 5
# _C.MODEL.SNAKE_HEAD.MSCORE_SNAKE_PERTURB = True
_C.MODEL.SNAKE_HEAD.MSCORE_SNAKE_LOSS_WEIGHT = 1.0

_C.MODEL.SNAKE_HEAD.PRE_OFFSET = False  # first snake also predict a global offset

_C.MODEL.SNAKE_HEAD.USE_ASPP = False
_C.MODEL.SNAKE_HEAD.ASPP_DIM = 64
_C.MODEL.SNAKE_HEAD.ASPP_DILATIONS = (1, 6, 12, 18)

_C.MODEL.SNAKE_HEAD.USE_PSP = False
_C.MODEL.SNAKE_HEAD.PSP_SIZE = (1, 2, 3, 6)

_C.MODEL.SNAKE_HEAD.LAST_UP_SAMPLE = False
_C.MODEL.SNAKE_HEAD.UP_SAMPLE_RATE = 2
_C.MODEL.SNAKE_HEAD.LAST_CHAMFER = False
_C.MODEL.SNAKE_HEAD.LAST_CHAMFER_WEIGHT = 5.0 / 3
_C.MODEL.SNAKE_HEAD.LAST_NEIGHBOR = False

_C.MODEL.SNAKE_HEAD.TRACK_PATH = False

_C.MODEL.SNAKE_HEAD.NEW_MATCHING = False

_C.MODEL.SNAKE_HEAD.INITIAL = "octagon"  # {"octagon", "box"};
_C.MODEL.SNAKE_HEAD.DE_LOC_TYPE = "derange"  # {"derange", "demean"}
_C.MODEL.SNAKE_HEAD.LOCAL_SPATIAL = False

_C.MODEL.SNAKE_HEAD.INDIVIDUAL_SCALE = False

_C.MODEL.SNAKE_HEAD.LOSS_TYPE = "smoothl1"  # {"smoothl1", "chamfer"}
_C.MODEL.SNAKE_HEAD.LOSS_ADAPTIVE = False
_C.MODEL.SNAKE_HEAD.LOSS_SEPARATE_REFINE = False
_C.MODEL.SNAKE_HEAD.LOSS_WEIGH = False
_C.MODEL.SNAKE_HEAD.LOSS_DISTRIBUTION = (1.0 / 3, 1.0 / 3, 2.0 / 3)
_C.MODEL.SNAKE_HEAD.LOSS_L1_BETA = 0.11
_C.MODEL.SNAKE_HEAD.EDGE_IN = False
_C.MODEL.SNAKE_HEAD.PRED_EDGE = False
_C.MODEL.SNAKE_HEAD.EDGE_IN_SEPARATE = (False, False)
_C.MODEL.SNAKE_HEAD.EDGE_POSITION = "before"  # {"before", "after"}
_C.MODEL.SNAKE_HEAD.DILATIONS = (1, 1)
_C.MODEL.SNAKE_HEAD.COORD_CONV = (False, False)
_C.MODEL.SNAKE_HEAD.EDGE_IN_TH = -1.0  # used for inference

_C.MODEL.SNAKE_HEAD.FILTER_WIDTH = 4

_C.MODEL.SNAKE_HEAD.USE_DEFORMABLE = (False, False)

_C.MODEL.SNAKE_HEAD.NUM_SAMPLING = 128
_C.MODEL.SNAKE_HEAD.MARK_INDEX = False
_C.MODEL.SNAKE_HEAD.REORDER_METHOD = "dsnake"  # {'dsnake', 'curvegcn'}
_C.MODEL.SNAKE_HEAD.JITTERING = 0.0
_C.MODEL.SNAKE_HEAD.POINT_WEIGH = False

_C.MODEL.SNAKE_HEAD.ATTENTION = False
_C.MODEL.SNAKE_HEAD.SELECTIVE_REFINE = False
_C.MODEL.SNAKE_HEAD.DOUBLE_SELECTIVE_REFINE = False


# utils
_C.MODEL.SNAKE_HEAD.VIS_PATH = False


# ---------------------------------------------------------------------------- #
# Edge Prediction Head (ZC)
# ---------------------------------------------------------------------------- #
_C.MODEL.EDGE_HEAD = CN()
_C.MODEL.EDGE_HEAD.NAME = "EdgeFPNHead"

_C.MODEL.EDGE_HEAD.TRAIN = True

_C.MODEL.EDGE_HEAD.IN_FEATURES = ["p2"]
_C.MODEL.EDGE_HEAD.STRONG_FEAT = False
# Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
# the correposnding pixel.
_C.MODEL.EDGE_HEAD.IGNORE_VALUE = 255
# Number of classes in the edge prediction head
_C.MODEL.EDGE_HEAD.NUM_CLASSES = 1  # (only foreground or not)
# Number of channels in the 3x3 convs inside semantic-FPN heads.
_C.MODEL.EDGE_HEAD.CONVS_DIM = 128
# Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
_C.MODEL.EDGE_HEAD.COMMON_STRIDE = 4
# Normalization method for the convolution layers. Options: "" (no norm), "GN".
_C.MODEL.EDGE_HEAD.NORM = "GN"
_C.MODEL.EDGE_HEAD.BCE_WEIGHT = (
    0  # 1:1 BCE harms the training, very small BCE not helpful
)

_C.MODEL.EDGE_HEAD.LOSS_WEIGHT = 1


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
