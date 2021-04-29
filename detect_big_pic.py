# coding=utf-8

"""this script is used to detect remote sensing images with large resolution"""

import time
import os
import torch
from torch.backends import cudnn
from collections import defaultdict
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from utils.utils_detect_big_pic import record_offset, add_offset, plot_one_box_new, py_nms

compound_coef = 0  # 更改使用模型的系数
force_input_size = None  # set None to use default size
path = 'detect_big_pic/ori_split'  # the path of the remote sensing image with large resolution
img_path = []
img_list = os.listdir(path)
print('检测路径下的图片为')
for i in range(len(img_list)):
    print(f'{img_list[i]}')

    img_path.append(os.path.join(path, img_list[i]))
print('-----------------------------------')


# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.3, 0.8), (0.9, 0.5)]  # 比例被改变了，与原始的不相同
# anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]  # 原始使用的长宽比
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.4  # for d1 threshold = 0.5, for d0 threshold = 0.2
iou_threshold = 0.2  # for d1 threshold = 0.6, for d0 threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['large-vehicle', 'small-vehicle']

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)  # seg is cut from ori image

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)

# model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
model.load_state_dict(torch.load('logs/' + 'birdview_vehicles_efficientdet-d0.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()

if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)  # get the results of the model

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

out = invert_affine(framed_metas, out)

offsets_list = record_offset(img_list)
print(f'图片左上角偏移量:{offsets_list}')
# print('------------------------------')

whole_list = []
print(f'共检测图片的个数:{len(out)}')
# for num in range(len(out)):
for num in range(len(out)):
    print(f'------------ {len(out[num]["rois"])} ---------------')
    for index in range(len(out[num]['rois'])):
        temp_list = []
        score = out[num]['scores'][index]
        class_ids = out[num]['class_ids'][index]
        split_loc = out[num]['rois'][index]
        print(f'******{offsets_list[num]}**********')
        ori_loc = add_offset(offsets_list[num], split_loc)
        for j in range(4):
            temp = ori_loc[j]
            temp_list.append(temp)
        temp_list.append(score)
        temp_list.append(class_ids)
        whole_list.append(temp_list)

print('--------------------------------------')
print(whole_list)
print(len(whole_list))

big_pic_path = r'detect_big_pic/ori_big'
img_lists = os.listdir(big_pic_path)
pic_path = os.path.join(big_pic_path, img_lists[0])
img = cv2.imread(pic_path)

# 此位置添加NMS算法 begin--------
whole_array = np.array(whole_list)
keeps = py_nms(whole_array, thresh=0.6)
print(f'****{keeps}******')  # 可以保留的predicted box的索引号
print(whole_list)
# print(f'........{whole_list[0]}')

seen_bboxes_index = []
seen_bboxes = []
for ii in range(len(keeps)):
    seen_bboxes_index.append(keeps[ii])
for jj in range(len(seen_bboxes_index)):
    index_order = seen_bboxes_index[jj]
    seen_bboxes.append(whole_list[index_order])
# end
print(f'---------->{seen_bboxes}')


for order in range(len(whole_list)):
    x1 = whole_list[order][0]
    y1 = whole_list[order][1]
    x2 = whole_list[order][2]
    y2 = whole_list[order][3]
    obj = obj_list[whole_list[order][5]]
    score = float(whole_list[order][4])
    detected_img = plot_one_box_new(img, [x1, y1, x2, y2], label=obj, score=score)

cv2.imwrite(f'detect_big_pic/result/big_picture.jpg', detected_img)

