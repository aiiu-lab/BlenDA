from yacs.config import CfgNode as CN
from numpy import pi

# CfgNode has two methods: merge_from_file and merge_from_list
_C = CN()

# ------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.DOMAIN_ATTENTION_NAMES = ['space_attn', 'instance_attn', 'channel_attn']
_C.TRAIN.DOMAIN_DISCRIMINATOR_NAMES= ['space_D', 'channel_D', 'instance_D']

_C.TRAIN.LR_ENCODER= 1e-05
_C.TRAIN.ENCODER_NAMES= ['encoder'] # set for a different learning rate

_C.TRAIN.LR_DECODER= 1e-05
_C.TRAIN.DECODER_NAMES= ['decoder']

_C.TRAIN.LR_MULTI_LABEL_CLASSIFIER= 1e-05
_C.TRAIN.MULTI_LABEL_CLASSIFIER_NAMES= ['multi_label_classifier']

_C.TRAIN.LR = 2e-4
_C.TRAIN.LR_BACKBONE_NAMES = ["backbone.0"]
_C.TRAIN.LR_BACKBONE = 2e-5
_C.TRAIN.LR_LINEAR_PROJ_NAMES = ['reference_points', 'sampling_offsets']
_C.TRAIN.LR_LINEAR_PROJ_MULT = 0.1
_C.TRAIN.LR_PROMPT = 0.0
_C.TRAIN.LR_PROMPT_NAMES = ['prompt']
_C.TRAIN.LR_HEAD = 2e-5
_C.TRAIN.LR_HEAD_NAMES = ['class_embed', 'bbox_embed']
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.EPOCHS = 50
_C.TRAIN.LR_DROP = 40
_C.TRAIN.LR_DROP_EPOCHS = None
_C.TRAIN.CLIP_MAX_NORM = 0.1 # gradient clipping max norm
_C.TRAIN.SGD = False # AdamW is used when setting this false
_C.TRAIN.FINETUNE = False

# ------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------
_C.MODEL = CN()

# Variants of Deformable DETR
_C.MODEL.WITH_BOX_REFINE = False
_C.MODEL.TWO_STAGE = False

# Model parameters
_C.MODEL.FROZEN_WEIGHTS = None # Path to the pretrained model. If set, only the mask head will be trained

# * Backbone
_C.MODEL.BACKBONE = 'resnet50' # Name of the convolutional backbone to use
_C.MODEL.DILATION = False # If true, we replace stride with dilation in the last convolutional block (DC5)
_C.MODEL.POSITION_EMBEDDING = 'sine' # ('sine', 'learned') Type of positional embedding to use on top of the image features
_C.MODEL.POSITION_EMBEDDING_SCALE = 2 * pi # position / size * scale
_C.MODEL.NUM_FEATURE_LEVELS = 4 # number of feature levels

# * Transformer
_C.MODEL.ENC_LAYERS = 6 # Number of encoding layers in the transformer
_C.MODEL.DEC_LAYERS = 6 # Number of decoding layers in the transformer
_C.MODEL.DIM_FEEDFORWARD = 1024 # Intermediate size of the feedforward layers in the transformer blocks
_C.MODEL.HIDDEN_DIM = 256 # Size of the embeddings (dimension of the transformer)
_C.MODEL.DROPOUT = 0.1 # Dropout applied in the transformer
_C.MODEL.NHEADS = 8 # Number of attention heads inside the transformer's attentions
_C.MODEL.NUM_QUERIES = 300 # Number of query slots
_C.MODEL.DEC_N_POINTS = 4
_C.MODEL.ENC_N_POINTS = 4

# * Segmentation
_C.MODEL.MASKS = False # Train segmentation head if the flag is provided

# * Domain Adaptation
_C.MODEL.BACKBONE_ALIGN = False
_C.MODEL.SPACE_ALIGN = False
_C.MODEL.CHANNEL_ALIGN = False
_C.MODEL.INSTANCE_ALIGN = False

# ------------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.AUX_LOSS = True # auxiliary decoding losses (loss at each layer)

# * Matcher
_C.LOSS.SET_COST_CLASS = 2. # Class coefficient in the matching cost
_C.LOSS.SET_COST_BBOX = 5. # L1 box coefficient in the matching cost
_C.LOSS.SET_COST_GIOU = 2. # giou box coefficient in the matching cost

# * Loss coefficients
_C.LOSS.MASK_LOSS_COEF = 1.
_C.LOSS.DICE_LOSS_COEF = 1.
_C.LOSS.CLS_LOSS_COEF = 2.
_C.LOSS.BBOX_LOSS_COEF = 5.
_C.LOSS.GIOU_LOSS_COEF = 2.
_C.LOSS.BACKBONE_LOSS_COEF = 0.1
_C.LOSS.SPACE_QUERY_LOSS_COEF = 0.1
_C.LOSS.CHANNEL_QUERY_LOSS_COEF = 0.1
_C.LOSS.INSTANCE_QUERY_LOSS_COEF = 0.1
_C.LOSS.FOCAL_ALPHA = 0.25
_C.LOSS.DA_GAMMA = 0

# ------------------------------------------------------------------------
# dataset parameters
# ------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DA_MODE = 'uda'
_C.DATASET.NUM_CLASSES = 9 # This should be set as max_class_id + 1
_C.DATASET.DATASET_FILE = 'cityscapes_to_foggy_cityscapes'
_C.DATASET.COCO_PATH = '../datasets'
_C.DATASET.COCO_PANOPTIC_PATH = None
_C.DATASET.REMOVE_DIFFICULT = False
_C.DATASET.CATEGORY_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

# ------------------------------------------------------------------------
# BlenDA
# ------------------------------------------------------------------------
_C.BLENDA = CN()
_C.BLENDA.MIXUP_SRC_INT_IMGS = False
_C.BLENDA.MIXUP_SRC_INT_DOMAIN_LABELS = False
_C.BLENDA.MIXUP_SRC_TGT_IMGS = False
_C.BLENDA.MIXUP_SRC_TGT_DOMAIN_LABELS = False
_C.BLENDA.INT_DOMAIN_LABEL = 0
_C.BLENDA.ALPHA = 5.0
_C.BLENDA.BETA = 1.0
_C.BLENDA.MIN_DELTA = 0.0

# ------------------------------------------------------------------------
# Distributed
# ------------------------------------------------------------------------
_C.DIST = CN()
_C.DIST.DISTRIBUTED = False
_C.DIST.RANK = None
_C.DIST.WORLD_SIZE = None
_C.DIST.GPU = None
_C.DIST.DIST_URL = None
_C.DIST.DIST_BACKEND = None

# ------------------------------------------------------------------------
# Miscellaneous
# ------------------------------------------------------------------------
_C.OUTPUT_DIR = '' # path where to save, empty for no saving
_C.DEVICE = 'cuda' # device to use for training / testing
_C.SEED = 42
_C.RESUME = '' # resume from checkpoint
_C.START_EPOCH = 0 # start epoch
_C.EVAL = False
_C.NUM_WORKERS = 2
_C.CACHE_MODE = False # whether to cache images on memory


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
