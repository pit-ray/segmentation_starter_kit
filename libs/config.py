from yacs.config import CfgNode

_C = CfgNode()
_C.DEVICE = 'cuda:0'
_C.NUM_WORKERS = 4
_C.RANDOM_SEED = None

_C.DATA = CfgNode()
_C.DATA.ROOT_DIR = './data'
_C.DATA.IMAGE_DIR = 'images'
_C.DATA.MASK_DIR = 'masks'
_C.DATA.IMG_EXT = '.png'

_C.DATA.BATCH_SIZE = 4
_C.DATA.IMG_WIDTH = 256
_C.DATA.IMG_HEIGHT = 256
_C.DATA.CLASS_RGB = [0, 255]

_C.MODEL = CfgNode()
_C.MODEL.NAME = 'Unet'
_C.MODEL.ENCODER_NAME = 'resnet18'

_C.TRAIN = CfgNode()
_C.TRAIN.MAX_EPOCH = 20
_C.TRAIN.LR = 1e-4


def get_cfg_defaults():
    return _C.clone()


def load_config(
        config_file: str,
        override_opts: list = [],
        freeze: bool = True) -> CfgNode:
    """
    Load config file.

    Args:
      config_file: the filename of yaml.
      override_opts: List of keys and values, such as [key1, val1, key2, val2]
      freeze: Whether the return value is unchangeable or not.
    """
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)

    if override_opts:
        cfg.merge_from_list(override_opts)

    if freeze:
        cfg.freeze()

    return cfg
