# coding=utf-8

# Author: Zylo117
# modified by Hs
#

# 批量检测图片(根据Kmeans，对anchor_ratios进行调整)
# 运行方式：
# Step 1:
# 该代码生成的文件在/home/fzh/Rotation-EfficinetDet/Evaluation_on_val_set/result_classname/文件夹下
# write_into_txt() 函数中path对应的路径下
#
# Step 2:
# 运行evaluation.py文件
#


# file tree:
# write_into_txt() 函数中path对应生成的txt文件所在位置
# path:代表将所要检测图片的basename存放到一个txt文件中
# imgpath:代表所要检测图片存放的路径 存放的图片是*.png格式
#
# line 239 write_into_txt(file_name, filecontent) 用于将检测结果写入到文件中
# line 241 display(filebasename, out, ori_imgs, imshow=False, imwrite=True) 用于保存检测结果图片

"""
Simple Inference Rotation Visualize of EfficientDet-D0-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from tqdm import tqdm
from matplotlib import colors
import matplotlib.pyplot as plt


def OPENCV2xywh(opencv_list):
    poly_list = []
    opencv_list[:5] = map(float, opencv_list[:5])
    x_c = int((opencv_list[0] + opencv_list[2]) / 2.)
    y_c = int((opencv_list[1] + opencv_list[3]) / 2.)
    width = int(opencv_list[2] - opencv_list[0])
    height = int(opencv_list[3] - opencv_list[1])
    theta = int(opencv_list[4])
    rect = ((x_c, y_c), (width, height), theta)
    poly = np.float32(cv2.boxPoints(rect))
    poly_list.append(poly)
    return poly_list


def write_into_txt(file_name, lists):
    path = r'/home/fzh/Rotation-EfficinetDet/Evaluation_on_val_set/result_classname'
    for idx in range(len(lists)):
        single_list = lists[idx]
        class_id = single_list[0]
        txt_name = file_name[class_id]
        txt_path = os.path.join(path, txt_name)

        with open(txt_path, 'a') as f_out:
            strline = str(single_list[1]) + ' ' + str(single_list[2]) + ' ' + str(single_list[3]) + \
                      ' ' + str(single_list[4]) + ' ' + str(single_list[5]) + ' ' + str(single_list[6]) + \
                      ' ' + str(single_list[7]) + ' ' + str(single_list[8]) + ' ' + str(single_list[9]) + \
                      ' ' + str(single_list[10]) + '\n'
            f_out.write(strline)


from backbone import EfficientDetBackbone
import cv2
import numpy as np
from efficientdet.Rotation_utils import Rotation_BBoxTransform, ClipBoxes, BBoxAddScores
from utils.Rotation_utils import eval_preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os
compound_coef = 0
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
# anchor_ratios = [(1.0, 1.0), (1.3, 0.8), (0.9, 0.5)]  # 第一次rotation训练用的anchor_ratios
# anchor_ratios = [(1.0, 1.0), (2, 2), (0.7, 1.4)]  # 原始使用的长宽比

anchor_ratios = [(1.0, 1.0), (0.4, 1.1), (0.7, 2.5), (1.8, 0.5)]
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

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)

# 模型读取水平框权重文件
# model.load_state_dict(torch.load(f'logs/birdview_vehicles_efficientdet-d0.pth', map_location='cpu'))

# 模型读取旋转框权重文件
model.load_state_dict(torch.load(f'/home/fzh/Rotation-EfficinetDet/logs/rotation_vehicles/efficientdet-d0_49_13500.pth',
                                 map_location='cpu'))

model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()

if use_float16:
    model = model.half()

path = r'/home/fzh/Rotation-EfficinetDet/Evaluation_on_val_set/imgnamefile.txt'
imgpath = r'/home/fzh/DOTA_devkit_YOLO-master/EfficientDet/images'
content = []
with open(path, 'r') as f_in:
    lines = f_in.readlines()
    for idx in range(len(lines)):
        line = lines[idx]
        line = line.strip().split(' ')
        content.append(line[0])

for i in tqdm(range(len(content)), ncols=88):
    filebasename = content[i]
    img_path = os.path.join(imgpath, filebasename + '.png')

    ori_imgs, framed_imgs, framed_metas = eval_preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)


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


    def display(filebasenme, preds, imgs, imshow=True, imwrite=False):
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
                cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{filebasename}.jpg', imgs[i])


    out = invert_affine(framed_metas, out)
    # print(f"out:{out}")

    file_name = ['Task1_large-vehicle.txt', 'Task1_small-vehicle.txt']
    rois = out[0]['rois']
    class_ids = out[0]['class_ids']
    scores = out[0]['scores']

    filecontent = []
    for ii in range(len(scores)):
        xmin, ymin, xmax, ymax, theta = rois[ii]
        rect = OPENCV2xywh([xmin, ymin, xmax, ymax, theta])[0].tolist()
        x1, y1 = float(rect[0][0]), float(rect[0][1])
        x2, y2 = float(rect[1][0]), float(rect[1][1])
        x3, y3 = float(rect[2][0]), float(rect[2][1])
        x4, y4 = float(rect[3][0]), float(rect[3][1])
        single_filecontent = [int(class_ids[ii]), filebasename, float(scores[ii]), x1, y1, x2, y2, x3, y3, x4, y4]
        filecontent.append(single_filecontent)

    write_into_txt(file_name, filecontent)
    #
    # display(filebasename, out, ori_imgs, imshow=False, imwrite=True)


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

