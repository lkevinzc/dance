from collections import OrderedDict
from torch.nn.functional import upsample
from deeplab_resnet import resnet101
from dextr_helper import *
import os
import torch
import numpy as np


class Dextr(object):
    def __init__(self, model_path='',
                 gpu_id=0, flip_test=True):
        if model_path == '':
            model_path = os.path.join(
                'cache', 'dextr_pascal-sbd.pth')
        self.pad = 50
        self.thres = 0.8
        self.device = torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
        self.flip_test = flip_test

        #  Create the network and load the weights
        self.net = resnet101(1, nInputChannels=4, classifier='psp')
        print("Initializing weights from: {}".format(model_path))
        state_dict_checkpoint = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        # Remove the prefix .module from the model when it is trained using DataParallel
        if 'module.' in list(state_dict_checkpoint.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in state_dict_checkpoint.items():
                name = k[7:]  # remove `module.` from multi-gpu training
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict_checkpoint
        self.net.load_state_dict(new_state_dict)
        self.net.eval()
        self.net.to(self.device)

    def segment(self, image, extreme_points_ori):
        #  Crop image to the bounding box from the extreme points and resize
        bbox = get_bbox(image, points=extreme_points_ori, pad=self.pad, zero_pad=True)
        crop_image = crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [
            self.pad,
            self.pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = input_dextr.transpose((2, 0, 1))[np.newaxis, ...]
        # import pdb; pdb.set_trace()
        if self.flip_test:
            inputs = np.concatenate([inputs, inputs[:, :, :, ::-1]], axis=0)
        inputs = torch.from_numpy(inputs)
        # Run a forward pass
        inputs = inputs.to(self.device)
        outputs = self.net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))
        outputs = outputs.data.numpy()
        if self.flip_test:
            outputs = (outputs[:1] + outputs[1:, :, :, ::-1]) / 2

        pred = np.transpose(outputs[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        result = crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=self.pad) > self.thres
        return result
