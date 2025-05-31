from typing import Dict
import numpy as np
import torch
import torchvision.utils as tvutils
import torch.nn.functional as F
import random
import colorsys


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: tuple((np.array(colorsys.hsv_to_rgb(*c))*255).astype(np.uint8)), hsv))
    random.shuffle(colors)
    return colors

ccc = random_colors(15)


def visualize_tensor(tensors_dict: Dict[str, torch.Tensor], mean, div):
    together = []

    rgb = unormalize(tensors_dict["rgb"].detach().cpu(), mean, div) * 255  # [n, c, h, w]
    rgb = torch.clamp(rgb, 0, 255).to(torch.uint8)

    mt = tensors_dict["mask_tensors"].detach().cpu().long()
    mt = F.one_hot(mt.squeeze(), 16).permute(0, 3, 1, 2).bool()  # [8, 256, 256, 1] ->> [8, 1, 256, 256]
    mt = mt[:, 1:]
    res = []
    for _rgb, _mask in zip(rgb, mt):
        # _mask[:] = False
        # for _i, _c in enumerate(ccc):
        #     _mask[_i, _i*10:_i*10+8] = True
        res.append(tvutils.draw_segmentation_masks(_rgb, _mask, alpha=0.7, colors=ccc))
    res = torch.stack(res, dim=0)
    together.append(res)

    labels = tensors_dict["pred_mask"].detach().cpu()
    # true_mask = tensors_dict["y"].detach().cpu().bool()
    labels = F.one_hot(labels.squeeze(), 16).permute(0, 3, 1, 2).bool()  # [8, 256, 256, 1] ->> [8, 1, 256, 256]
    labels = labels[:, 1:]
    res = []
    for _rgb, _lab in zip(rgb, labels):
        # _img = tvutils.draw_segmentation_masks(_rgb, _mask, alpha=0.7, colors="green")
        res.append(tvutils.draw_segmentation_masks(_rgb, _lab, alpha=0.7, colors=ccc))
    res = torch.stack(res, dim=0)
    together.append(res)

    labels = tensors_dict["target"].detach().cpu()
    # true_mask = tensors_dict["y"].detach().cpu().bool()
    labels = F.one_hot(labels.squeeze(), 16).permute(0, 3, 1, 2).bool()  # [8, 256, 256, 1] ->> [8, 1, 256, 256]
    labels = labels[:, 1:]
    res = []
    for _rgb, _lab in zip(rgb, labels):
        # _img = tvutils.draw_segmentation_masks(_rgb, _mask, alpha=0.7, colors="green")
        res.append(tvutils.draw_segmentation_masks(_rgb, _lab, alpha=0.7, colors=ccc))
    res = torch.stack(res, dim=0)
    together.append(res)

    if len(together) == 0:
        return None
    together = torch.cat(together, dim=3)
    # together = together.permute(1,0,2,3).contiguous()
    # together = together.view(together.size(0), -1, together.size(3))
    together = tvutils.make_grid(together, nrow=1, padding=6)
    return together.numpy().astype(np.uint8)



def mask_to_color(mask):
    color_map = {
        0: (0, 0, 0),  # 黑色
        1: (0, 255, 0),  # 绿色
        2: (0, 0, 255),  # 蓝色
        3: (255, 255, 0),  # 黄色
        4: (255, 0, 255),  # 紫色
        5: (0, 255, 255),  # 青色
        6: (128, 0, 0),  # 深红色
        7: (0, 128, 0),  # 深绿色
        8: (0, 0, 128),  # 深蓝色
        9: (128, 128, 0),  # 深黄色
        10: (128, 0, 128),  # 深紫色
        11: (0, 128, 128),  # 深青色
        12: (192, 192, 192),  # 灰色
        13: (255, 128, 0),  # 橙色
        14: (128, 255, 0),  # 浅绿色
        15: (0, 128, 255)}  # 浅蓝色

    # 创建一个彩色空间的 mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    for x in range(mask.shape[0]):
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                color_mask[x, i, j] = color_map[mask[x, i, j]]
    return color_mask


def unormalize(tensor, mean, div):
    for c, (m, d) in enumerate(zip(mean, div)):
        tensor[:,c,:,:].mul_(d).add_(m)
    return tensor
