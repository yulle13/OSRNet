
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import Colors
import torch
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import sys
from PIL import Image
import pycocotools.mask as maskUtils
import utils
import inference as infer
from lib.dataset.reader import KWOBDataset
from demo_utils import *
import glob


def recover_amodal(config_file, load_model, image, image_name, in_bboxes, in_modal):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    recovernet = RecoverModal(config_file, load_model)

    # read seg-model-prediction
    ret_modal = []
    ret_bboxes = []
    _pre_modal0 = in_modal.cpu().numpy() #np.array(ret_modal)
    _pre_bboxes0 = in_bboxes.xyxy.cpu().numpy()  #np.array(ret_bboxes)
    for box, mask in zip(_pre_bboxes0, _pre_modal0):
        ret_modal.append(mask.data.squeeze(0))
        ret_bboxes.append(box)
    _pre_bboxes = np.array(ret_bboxes)
    _pre_modal = np.array(ret_modal)
        
    # transform xyxy to xywh
    _pre_bboxes = utils.xyxy_to_xywh(_pre_bboxes)
    # 调整顺序
    match_pre = utils.sort_box(_pre_bboxes)
    pre_modal = np.empty_like(_pre_modal)
    pre_bboxes = np.empty_like(_pre_bboxes)

    for old_index, new_index in match_pre.items():
        pre_modal[new_index] = _pre_modal[old_index]
        pre_bboxes[new_index] = _pre_bboxes[old_index]

    # seg-model-prediction pass recover-model to amodal
    big_boxes = expand_bbox(pre_bboxes, enlarge_ratio=3., single_ratio=1.5)
    amodal_patches_pred = infer.infer_amodal(recovernet.model, image, pre_modal, big_boxes)
    amodal_recover = infer.patch_to_fullimage(
        amodal_patches_pred, big_boxes, pre_modal.shape[1], pre_modal.shape[2], interp='linear')

    image_show = image.copy()
    exclude = []
    selidx = np.array([i for i in np.arange(big_boxes.shape[0]) if i not in exclude])
    colors = [(np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0] for i in range(len(selidx))]
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(image_show)
    plt.axis('off')
    pface, pedge = polygon_drawing(amodal_recover, selidx, colors, bbox=None, thickness=3)
    ax.add_collection(pface)
    ax.add_collection(pedge)
    plt.savefig('./result/pre_all_amodal.png')
    plt.close()


    
     # save modal and amodal
    for x in range(len(amodal_recover)):
        image_show = image.copy()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(pre_modal[x], cmap='binary')
        # plt.axis('off')
        # plt.savefig('./result/pre_modal{}.png'.format(x))
        # plt.close()
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.imshow(image_show)
        plt.axis('off')
        mask = np.ma.masked_where(amodal_recover[x] == 0, amodal_recover[x])
        plt.imshow(mask, cmap='spring', alpha=0.8)
        plt.savefig('./result/pre_amodal{}.png'.format(x))
        plt.close()


    #  ordering
    order_matrix = infer.infer_gt_order(pre_modal, amodal_recover)
    grab_order = generate_grasp_order(order_matrix)
    match_order = {}
    for s,j in enumerate(grab_order):
        match_order[j] = s
    
    
    for i, amodal_re in enumerate(amodal_recover):
        contours, _ = cv2.findContours(amodal_re, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(max_contour)
        center, axes, angle = ellipse
        minor_axis_length = min(axes[0], axes[1])
        grasp_length = minor_axis_length * 1.1
        grasp_width = 50

        rect = ((center[0], center[1]), (grasp_length, grasp_width), angle)
        box = cv2.boxPoints(rect)
        box = np.int_(box)

        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # 字体大小
        color = (255, 0, 0)  # 文本颜色 (B, G, R)
        thickness = 2  # 字体粗细
        line_type = cv2.LINE_AA  # 线条类型
        cv2.putText(image, str(match_order[i]), (box[3]), font, font_scale, color, thickness, line_type)

    plt.figure(figsize=(16 , 16))
    plt.title('Grasp rect')    
    plt.imshow(image)
    plt.savefig('./result/Grasp.png')
    plt.close()



if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    img_name = 'IMG_20240415_194543.jpg' 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='./weights/best.pt', help="Path to ONNX model")
    parser.add_argument("--source", type=str,
                        default='/{}'.format(img_name),
                        help="Path to input image") #/home/zxy/zxy_project/ultralytics_0/data/kwob_yolo/train/images
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--config", type=str, default='./configs/config_kwob.yaml')
    parser.add_argument("--load_model", type=str, default='./weights/ckp60.pth')
    parser.add_argument("--use_camera", type=bool, default=True, help="Use camera as input source")

    args = parser.parse_args()
    # Get the last image in the folder
    # image_folder = '/home/zxy/try_data/'
    # image_files = sorted(glob.glob(f"{image_folder}*.jpg"))
    # if not image_files:
    #     raise FileNotFoundError("No images found in the specified folder.")
    # img_name = os.path.basename(image_files[-1])
    # args.source = os.path.join(image_folder, img_name)
    # Build model
    
    model = YOLO(args.model)


    # # Read image by OpenCV
    img = cv2.imread(args.source)
    # Get the original dimensions of the image
    h, w, _ = img.shape

    # Determine the scale factor and new dimensions
    if h < w:
        scale = 640 / h
        new_h, new_w = 640, int(w * scale)
    else:
        scale = 640 / w
        new_h, new_w = int(h * scale), 640

    # Resize the image while maintaining the aspect ratio
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Crop the center 640x640 region
    start_x = (new_w - 640) // 2
    start_y = (new_h - 640) // 2
    img = img[start_y:start_y + 640, start_x:start_x + 640]
    # Run inference on the source
    results = model(img, conf=args.conf)  # list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        result.save(filename="./result/result.jpg")  # save to disk
    
    # # # Draw bboxes and polygons
    if len(boxes) > 0:
        # model.draw_and_visualize(image, boxes, masks)
        recover_amodal(args.config, args.load_model, img, img_name, boxes, masks)