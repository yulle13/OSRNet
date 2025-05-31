import numpy as np
import cv2
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import utils
from . import reader
import time


class PartialCompDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config["dataset"]
        if self.dataset == "COCOA":
            self.data_reader = reader.COCOADataset(config["{}_annot_file".format(phase)])
        elif self.dataset == "KINS":
            self.data_reader = reader.KINSLVISDataset(self.dataset, config["{}_annot_file".format(phase)])
        elif self.dataset == "KWOB":
            self.data_reader = reader.KWOBDataset(config["{}_annot_file".format(phase)])
        else:
            raise NotImplementedError

        self.img_transform = transforms.Compose([transforms.Normalize(config["data_mean"], config["data_std"])])
        self.sz = config["input_size"]
        self.phase = phase
        self.config = config
        self.eraser_setter = utils.EraserSetter(config["eraser_setter"])
        self.eraser_front_prob = config["eraser_front_prob"]

    def __len__(self):
        return self.data_reader.get_instance_length()

    def _load_image(self, fn):
        return Image.open(fn).convert("RGB")

    def __getitem__(self, idx):
        modal, bbox, category, imgfn, amodal = self.data_reader.get_instance(idx)
        # if self.dataset == "KINS":
        #     modal *= category
        #     amodal *= category
        if self.phase == 'train':
            randidx = np.random.choice(len(self))
            eraser, _, _, _, _ = self.data_reader.get_instance(randidx)
            eraser = self.eraser_setter(modal, eraser)  # uint8 {0, 1}
            erased_modal = modal.copy().astype(np.float32)
            if np.random.rand() < self.eraser_front_prob:
                erased_modal[eraser == 1] = 0  # eraser above modal
            modal = erased_modal
        # print(f"dataset time {time.time() - t_start}")

        centerx = bbox[0] + bbox[2] / 2.0
        centery = bbox[1] + bbox[3] / 2.0
        size = max([np.sqrt(bbox[2] * bbox[3] * self.config["enlarge_box"]), bbox[2] * 1.1, bbox[3] * 1.1])
        if size < 5 or np.all(modal == 0):
            return self.__getitem__(np.random.choice(len(self)))

        # shift & scale aug
        if self.phase == "train":
            centerx += np.random.uniform(*self.config["base_aug"]["shift"]) * size
            centery += np.random.uniform(*self.config["base_aug"]["shift"]) * size
            size /= np.random.uniform(*self.config["base_aug"]["scale"])

        # crop
        new_bbox = [int(centerx - size * 0.6), int(centery - size * 0.6), int(size * 1.2), int(size * 1.2)]
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)), (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        amodal = cv2.resize(utils.crop_padding(amodal, new_bbox, pad_value=(0,)), (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        flip = False
        if self.phase == "train":
            if self.config["base_aug"]["flip"] and np.random.rand() > 0.5:
                flip = True
                modal = modal[:, ::-1]
                amodal = amodal[:, ::-1]
            else:
                flip = False

        modal = torch.from_numpy(modal.astype(np.float32)).unsqueeze(0)  # [1,256,256]
        # modal = torch.from_numpy(modal.astype(np.int_))
        amodal = torch.from_numpy(amodal.astype(np.int_))

        rgb = np.array(self._load_image(os.path.join(self.config["{}_image_root".format(self.phase)], imgfn)))  # uint8
        rgb = cv2.resize(utils.crop_padding(rgb, new_bbox, pad_value=(0, 0, 0)), (self.sz, self.sz), interpolation=cv2.INTER_CUBIC)
        if self.phase == 'train':
            if self.config['HSV'] > np.random.rand():
                hsv = RandomHSV()
                rgb = hsv(rgb)
        if flip:
            rgb = rgb[:, ::-1, :]
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.0)
        rgb = self.img_transform(rgb)  # CHW

        return modal, rgb, amodal


class RandomHSV:
    """
    This class is responsible for performing random adjustments to the Hue, Saturation, and Value (HSV) channels of an
    image.

    The adjustments are random but within limits set by hgain, sgain, and vgain.
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img):
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return img
