# coding=utf-8

# Author: Zylo117
# modified by Hs
# 检测单张图片


"""
Simple Inference Rotation Visualize of EfficientDet-D0-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import matplotlib.pyplot as plt

from backbone import EfficientDetBackbone
import cv2
import numpy as np
from efficientdet.Rotation_utils import Rotation_BBoxTransform, ClipBoxes, BBoxAddScores
from utils.Rotation_utils import eval_preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

compound_coef = 0
force_input_size = None  # set None to use default size

img_path = 'test/P0454_954_412.jpg'
# merge images
# img_path = r'/home/fzh/Rotation-EfficinetDet/detect_big_pic_obb/SplitImages/P0116_89_412.jpg'

# replace this part with your project's anchor config
# anchor_ratios = [(1.0, 1.0), (1.3, 0.8), (0.9, 0.5)]  # not using Kmeans

anchor_ratios = [(1.0, 1.0), (0.4, 1.1), (0.7, 2.5), (1.8, 0.5)]  # using Kmeans
# anchor_ratios = [(1.0, 1.0), (2, 2), (0.7, 1.4)]  # 原始使用的长宽比
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.6
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True
'''
obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']
'''

obj_list = ['large-vehicle', 'small-vehicle']

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = eval_preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)

# 模型读取水平框权重文件
# model.load_state_dict(torch.load(f'logs/birdview_vehicles_efficientdet-d0.pth', map_location='cpu'))

# 模型读取旋转框权重文件
model.load_state_dict(torch.load(f'/home/fzh/REfficientDetv1/logs/efficientdet-d0_49_13500.pth',
                                 map_location='cpu'))

model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()

if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = Rotation_BBoxTransform()
    clipBoxes = ClipBoxes()
    addBoxes = BBoxAddScores()
    # threshold 用于分类得分的阈值，iou_threshold用于skew iou计算的阈值
    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes, addBoxes,
                      threshold, iou_threshold)


# def OPENCV2xywh(opencv_list):
#     poly_list = []
#     for idx in range(len(opencv_list)):
#         opencv_list[idx][:5] = map(float, opencv_list[idx][:5])
#         x_c, y_c = int(opencv_list[idx][0]), int(opencv_list[idx][1])
#         width, height = int(opencv_list[idx][2]), int(opencv_list[idx][3])
#         theta = int(opencv_list[idx][4])
#         rect = ((x_c, y_c), (width, height), theta)
#         poly = np.float32(cv2.boxPoints(rect))
#         poly_list.append(poly)
#     return poly_list

def OPENCV2xywh(opencv_list):
    poly_list = []
    opencv_list[:5] = map(float, opencv_list[:5])
    # x_c, y_c = int(opencv_list[0]), int(opencv_list[1])
    x_c = int((opencv_list[0] + opencv_list[2]) / 2.)
    y_c = int((opencv_list[1] + opencv_list[3]) / 2.)
    # width, height = int(opencv_list[2]), int(opencv_list[3])
    width = int(opencv_list[2] - opencv_list[0])
    height = int(opencv_list[3] - opencv_list[1])
    theta = int(opencv_list[4])
    rect = ((x_c, y_c), (width, height), theta)
    poly = np.float32(cv2.boxPoints(rect))
    poly_list.append(poly)
    return poly_list


def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            """
            preds[i]['rois'][j] = [xmin, ymin, xmax, ymax, theta]
            """
            xmin, ymin, xmax, ymax, theta = preds[i]['rois'][j].astype(np.float)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            # ori code Zylo117
            # plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score, color=color_list[get_index_label(obj, obj_list)])
            if theta <= -90:
                color = [0, 255, 0]
            else:
                color = [0, 0, 255]
            # rotation detection code
            rect = OPENCV2xywh([xmin, ymin, xmax, ymax, theta])
            rect = np.int0(rect)
            rect = np.array(rect)

            # 绘图使用cv2.drawContours
            cv2.drawContours(
                image=imgs[i],
                contours=rect,
                contourIdx=-1,
                color=color,
                thickness=2
            )

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)
            # imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
            # plt.imshow(imgs[i])
            # plt.show()

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


out = invert_affine(framed_metas, out)
display(out, ori_imgs, imshow=False, imwrite=True)


# 运行10次输出
# print('running speed test...')
# with torch.no_grad():
#     print('test1: model inferring and postprocessing')
#     print('inferring image for 10 times...')
#     t1 = time.time()
#     for _ in range(10):
#         _, regression, classification, anchors = model(x)
#
#         out = postprocess(x,
#                           anchors, regression, classification,
#                           regressBoxes, clipBoxes,
#                           threshold, iou_threshold)
#         out = invert_affine(framed_metas, out)
#
#     t2 = time.time()
#     tact_time = (t2 - t1) / 10
#     print(f'detecting {img_path} {tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

