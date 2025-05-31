import json
import numpy as np
import sys
from PIL import Image
sys.path.append('.')
import cvbase as cvb
import pycocotools.mask as maskUtils
import utils
import cv2
def read_COCOA(ann, h, w):
    if 'visible_mask' in ann.keys():
        rle = [ann['visible_mask']]
    else:
        rles = maskUtils.frPyObjects([ann['segmentation']], h, w)
        rle = maskUtils.merge(rles)
    modal = maskUtils.decode(rle).squeeze()
    if np.all(modal != 1):
        # if the object if fully occluded by others,
        # use amodal bbox as an approximated location,
        # note that it will produce random amodal results.
        amodal = maskUtils.decode(maskUtils.merge(
            maskUtils.frPyObjects([ann['segmentation']], h, w)))
        bbox = utils.mask_to_bbox(amodal)
    else:
        bbox = utils.mask_to_bbox(modal)
    return modal, bbox, 1 # category as constant 1


class COCOADataset(object):

    def __init__(self, annot_fn):
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']

        self.indexing = []
        for i, ann in enumerate(self.annot_info):
            for j in range(len(ann['regions'])):
                self.indexing.append((i, j))

    def get_instance_length(self):
        return len(self.indexing)

    def get_image_length(self):
        return len(self.images_info)

    def get_gt_ordering(self, imgidx):
        num = len(self.annot_info[imgidx]['regions'])
        gt_order_matrix = np.zeros((num, num), dtype=np.int)
        order_str = self.annot_info[imgidx]['depth_constraint']
        if len(order_str) == 0:
            return gt_order_matrix
        order_str = order_str.split(',')
        for o in order_str:
            idx1, idx2 = o.split('-')
            idx1, idx2 = int(idx1) - 1, int(idx2) - 1
            gt_order_matrix[idx1, idx2] = 1
            gt_order_matrix[idx2, idx1] = -1
        return gt_order_matrix # num x num

    def get_instance(self, idx):
        imgidx, regidx = self.indexing[idx]
        # img
        img_info = self.images_info[imgidx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        # region
        reg = self.annot_info[imgidx]['regions'][regidx]
        modal, bbox, category = read_COCOA(reg, h, w)
        amodal = maskUtils.decode(maskUtils.merge(
                maskUtils.frPyObjects([reg['segmentation']], h, w)))

        return modal, bbox, category, image_fn, amodal

    def get_image_instances(self, idx, with_gt=False, with_anns=False, ignore_stuff=False):
        ann_info = self.annot_info[idx]
        img_info = self.images_info[idx]
        image_fn = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        for reg in ann_info['regions']:
            if ignore_stuff and reg['isStuff']:
                continue
            modal, bbox, category = read_COCOA(reg, h, w)
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            if with_gt:
                amodal = maskUtils.decode(maskUtils.merge(
                    maskUtils.frPyObjects([reg['segmentation']], h, w)))
                ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn, ann_info
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn


def mask_to_polygon(mask, tolerance=1.0, area_threshold=1):
    """Convert object's mask to polygon [[x1,y1, x2,y2 ...], [...]]
    Args:
        mask: object's mask presented as 2D array of 0 and 1
        tolerance: maximum distance from original points of polygon to approximated
        area_threshold: if area of a polygon is less than this value, remove this small object
    """
    from skimage import measure
    polygons = []
    # pad mask with 0 around borders
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_mask, 0.5)
    # Fix coordinates after padding
    contours = np.subtract(contours, 1)
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) > 2:
            contour = np.flip(contour, axis=1)
            reshaped_contour = []
            for xy in contour:
                reshaped_contour.append(xy[0])
                reshaped_contour.append(xy[1])
            reshaped_contour = [point if point > 0 else 0 for point in reshaped_contour]

            # Check if area of a polygon is enough
            rle = maskUtils.frPyObjects([reshaped_contour], mask.shape[0], mask.shape[1])
            area = maskUtils.area(rle)
            if sum(area) > area_threshold:
                polygons.append(reshaped_contour)
    return polygons

def read_KINS(ann):
    modal = maskUtils.decode(ann['inmodal_seg']) # HW, uint8, {0, 1}
    bbox = ann['inmodal_bbox'] # luwh
    category = ann['category_id']
    if 'score' in ann.keys():
        score = ann['score']
    else:
        score = 1.
    return modal, bbox, category, score


class KINSLVISDataset(object):

    def __init__(self, dataset, annot_fn):
        self.dataset = dataset
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']
        self.category_info = data['categories']

        # make dict
        self.imgfn_dict = dict([(a['id'], a['file_name']) for a in self.images_info])
        self.size_dict = dict([(a['id'], (a['width'], a['height'])) for a in self.images_info])
        self.anns_dict = self.make_dict()
        self.img_ids = list(self.anns_dict.keys())

    def get_instance_length(self):
        return len(self.annot_info)

    def get_image_length(self):
        return len(self.img_ids)
    
    def get_instance(self, idx):
        ann = self.annot_info[idx]
        # img
        imgid = ann['image_id']
        w, h = self.size_dict[imgid]
        image_fn = self.imgfn_dict[imgid]
        # instance
        
        modal, bbox, category, _ = read_KINS(ann)
    
        amodal = maskUtils.decode(
                maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
        return modal, bbox, category, image_fn, amodal

    def make_dict(self):
        anns_dict = {}
        for ann in self.annot_info:
            image_id = ann['image_id']
            if not image_id in anns_dict:
                anns_dict[image_id] = [ann]
            else:
                anns_dict[image_id].append(ann)
        return anns_dict # imgid --> anns

    def get_image_instances(self, idx, with_gt=False, with_anns=False):
        imgid = self.img_ids[idx]
        image_fn = self.imgfn_dict[imgid]
        w, h = self.size_dict[imgid]
        anns = self.anns_dict[imgid]
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []
        #ret_score = []
        for ann in anns:
            if self.dataset == 'KINS':
                modal, bbox, category, score = read_KINS(ann)
            else:
                raise Exception("No such dataset: {}".format(self.dataset))
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            #ret_score.append(score)
            if with_gt:
                amodal = maskUtils.decode(
                    maskUtils.frPyObjects(ann['segmentation'], h, w)).squeeze()
                ret_amodal.append(amodal)
        if with_anns:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn, anns
        else:
            return np.array(ret_modal), np.array(ret_category), np.array(ret_bboxes), np.array(ret_amodal), image_fn


def read_KWOB(ann):
    modal = maskUtils.decode(ann['segmentation']) # HW, uint8, {0, 1}
    bbox = ann['bbox'] # luwh
    category = ann['category_id']
    return modal, bbox, category



class KWOBDataset(object):

    def __init__(self, annot_fn):
        
        data = cvb.load(annot_fn)
        self.images_info = data['images']
        self.annot_info = data['annotations']
        self.category_info = data['categories']

        # make dict
        self.imgfn_dict = dict([(a['id'], a['file_name']) for a in self.images_info])
        self.size_dict = dict([(a['id'], (a['width'], a['height'])) for a in self.images_info])
        self.anns_dict = self.make_dict()
        self.img_ids = list(self.anns_dict.keys())

    def get_instance_length(self):
        return len(self.annot_info)

    def get_image_length(self):
        return len(self.img_ids)

    def get_gt_ordering(self, num, imgidx):  # 返回遮挡关系

        gt_order_matrix = np.zeros((num, num), dtype=np.int_)
        if self.images_info[imgidx]['occlusion_relation'] == '':
            return gt_order_matrix
        order_str = self.images_info[imgidx]['occlusion_relation']
        order_str = order_str.split(',')
        for o in order_str:
            if o != '':
                idx1, idx2 = o.split('-')
                gt_order_matrix[int(idx1), int(idx2)] = 1
                gt_order_matrix[int(idx2), int(idx1)] = -1
        return gt_order_matrix  # num x num
    
    def get_instance(self, idx):
        ann = self.annot_info[idx]
        # img
        imgid = ann['image_id']
        w, h = self.size_dict[imgid]
        image_fn = self.imgfn_dict[imgid]
        # instance
        modal, bbox, category = read_KWOB(ann)
        amodal = maskUtils.decode(ann['amodal_seg']).squeeze()
        amodal = np.logical_or(amodal, modal).astype(np.uint8)
        # amodal = maskUtils.decode(
        #         maskUtils.frPyObjects(ann['amodal_seg'], h, w)).squeeze()  # 改为 amodal and modal
        return modal, bbox, category, image_fn, amodal

    def make_dict(self):
        anns_dict = {}
        for ann in self.annot_info:
            image_id = ann['image_id']
            if not image_id in anns_dict:
                anns_dict[image_id] = [ann]
            else:
                anns_dict[image_id].append(ann)
        return anns_dict # imgid --> anns

    def get_image_instances(self, idx):
        idx = self.find_keys_by_value(idx, self.imgfn_dict)
        idx = idx[0]
        imgid = self.img_ids[idx]
        image_fn = self.imgfn_dict[imgid]
        w, h = self.size_dict[imgid]
        anns = self.anns_dict[imgid]
        ret_modal = []
        ret_bboxes = []
        ret_category = []
        ret_amodal = []

        for i in range(len(anns)):
            # 按实例顺序读取
            for ann in anns:
                if i == ann["inst_id"]:
                    modal, bbox, category = read_KWOB(ann)
                    break
                
            ret_modal.append(modal)
            ret_bboxes.append(bbox)
            ret_category.append(category)
            amodal = maskUtils.decode(ann['amodal_seg']).squeeze()
            amodal = np.logical_or(amodal, modal).astype(np.uint8)
            ret_amodal.append(amodal)
        order_image = self.get_gt_ordering(len(ret_modal), imgid)

        return np.array(ret_modal), np.array(ret_bboxes), np.array(ret_amodal), image_fn, order_image ,w, h
    
    def find_keys_by_value(self, target_value, data):
        return [key for key, value in data.items() if value == target_value]
# def get_modal(label, h, w):
#     import re
#     our_class = {
#     "Plastic bottle": 1,
#     "Plastic box": 2,
#     "Paper box": 3,
#     "Paper cup": 4,
#     "Metal bottle": 5,
#     "Metal box": 6,
#     "Glass bottle": 7,
#     "Glass cup": 8,
#     'Plastic cup': 9,
#     "Wooden": 10,
#     "Cloth": 11,
#     "Shoes": 12,
#     "Book": 13}

#     points = label['points']
#     _label = label['label']
#     # 将points转成seg, 得到对于的modal
#     poly = np.array(points).reshape(-1, 2)
#     segmentation = [poly.flatten().tolist()]
#     mask = np.zeros((h, w), dtype=np.uint8)
#     seg = np.array(segmentation, dtype=np.int32)
#     seg = seg.reshape((-1, 1, 2))
#     _class = re.search(r'_(.*?)_', _label)
#     if _class.group(1) in our_class:
#         category = our_class[_class.group(1)]
#     else:
#         print('标签类别错误')
#         print(_label)
#         category = 0
#     modal = cv2.fillPoly(mask, [seg], 1)
#     return modal, category


def mask_to_bbox(mask):
    mask = (mask != 0)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()] # xywh