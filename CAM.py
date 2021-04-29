# coding=utf-8
import os
import torch
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import eval_preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


compound_coef = 0  # 更改使用模型的系数
force_input_size = None  # set None to use default size
img_path = 'test/1250.jpg'
# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.3, 0.8), (0.9, 0.5)]  # 比例被改变了，与原始的不相同
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2  # for d1 threshold = 0.5, for d0 threshold = 0.2
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

ori_imgs, framed_imgs, framed_metas = eval_preprocess(img_path, max_size=input_size)  # seg is cut from ori image

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

print(out)
"""
out = 
[ {
    'rois' : [ [x1, y1, x2, y2], [x1, y1, x2, y2]...]
    'class_ids' : [1, 0]
    'scores: [score1, score1]
  }
]
"""

scores = out[0]['scores']