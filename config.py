from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
# config.TRAIN.batch_size = 8 # [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training
config.TRAIN.batch_size = 12 # [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G

config.TRAIN.n_epoch_init = 5000
# config.TRAIN.n_epoch_init = 100
config.TRAIN.lr_decay_init = 0.1
config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 5000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
# config.TRAIN.hr_img_path = 'DIV2K/DIV2K_train_HR/'
# config.TRAIN.hr_img_path = 'Middlebury/Train/HR/' # only single channel is used
config.TRAIN.hr_img_path = 'Middlebury/Train/HR' # only single channel is used
config.TRAIN.rgb_img_path = 'Middlebury/Train/RGB'
# config.TRAIN.lr_img_path = 'DIV2K/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'Middlebury/Valid/HR'
config.VALID.lr_img_path = 'Middlebury/Valid/LR'
config.VALID.rgb_img_path = 'Middlebury/Valid/RGB'
# config.VALID.lr_img_path = 'DIV2K/DIV2K_valid_LR_bicubic/X4/'
# config.VALID.rgb_img_path = 'DIV2K/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
