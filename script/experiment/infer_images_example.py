"""Example
python script/experiment/infer_images_example.py \
--model_weight_file YOUR_MODEL_WEIGHT_FILE
"""
from __future__ import print_function

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable

import numpy as np
import argparse
import cv2
from PIL import Image
import os.path as osp

from tri_loss.model.Model import Model
from tri_loss.utils.utils import load_state_dict
from tri_loss.utils.utils import set_devices
from tri_loss.utils.dataset_utils import get_im_names
from tri_loss.utils.distance import normalize


class Config(object):
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
        parser.add_argument('--last_conv_stride', type=int, default=1, choices=[1, 2])
        parser.add_argument('--ckpt_file', type=str, default='')
        parser.add_argument('--model_weight_file', type=str, default='')

        args = parser.parse_args()

        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        # Image Processing
        self.resize_h_w = args.resize_h_w
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        # The last block of ResNet has stride 2. We can set the stride to 1 so that
        # the spatial resolution before global pooling is doubled.
        self.last_conv_stride = args.last_conv_stride

        # This contains both model weight and optimizer state
        self.ckpt_file = args.ckpt_file
        # This only contains model weight
        self.model_weight_file = args.model_weight_file


def pre_process_im(im, cfg):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""

    # Resize.
    im = cv2.resize(im, cfg.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    # scaled by 1/255.
    im = im / 255.

    # Subtract mean and scaled by std
    im = im - np.array(cfg.im_mean)
    im = im / np.array(cfg.im_std).astype(float)

    # shape [H, W, 3] -> [1, 3, H, W]
    im = im.transpose(2, 0, 1)[np.newaxis]
    return im


def main():
    cfg = Config()

    TVT, TMO = set_devices(cfg.sys_device_ids)

    #########
    # Model #
    #########

    model = Model(last_conv_stride=cfg.last_conv_stride)
    # Set eval mode. Force all BN layers to use global mean and variance, also disable dropout.
    model.eval()
    # Transfer Model to Specified Device.
    TMO([model])

    #####################
    # Load Model Weight #
    #####################

    used_file = cfg.model_weight_file or cfg.ckpt_file
    loaded = torch.load(used_file, map_location=(lambda storage, loc: storage))
    if cfg.model_weight_file == '':
        loaded = loaded['state_dicts'][0]
    load_state_dict(model, loaded)
    print('Loaded model weights from {}'.format(used_file))

    ###################
    # Extract Feature #
    ###################

    im_dir = osp.expanduser('~/Dataset/market1501/Market-1501-v15.09.15/query')
    im_paths = get_im_names(im_dir, pattern='*.jpg', return_path=True, return_np=False)

    all_feat = []
    for i, im_path in enumerate(im_paths):
        im = np.asarray(Image.open(im_path).convert('RGB'))
        im = pre_process_im(im, cfg)
        im = Variable(TVT(torch.from_numpy(im).float()), volatile=True)
        feat = model(im)
        feat = feat.data.cpu().numpy()
        all_feat.append(feat)
        if (i + 1) % 100 == 0:
            print('{}/{} images done'.format(i, len(im_paths)))
    all_feat = np.concatenate(all_feat)
    print('all_feat.shape:', all_feat.shape)
    all_feat = normalize(all_feat, axis=1)
    # You can save your im_paths and features, or do anything else ...


if __name__ == '__main__':
    main()
