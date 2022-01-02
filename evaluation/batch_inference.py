"""This script is used to inference batch images and get all detection results on val set."""
# coding=utf-8

import torch
from tqdm import tqdm
from backbone import EfficientDetBackbone
import cv2
import numpy as np
from efficientdet.rotation_utils import Rotation_BBoxTransform, ClipBoxes, BBoxAddScores
from utils.rotation_utils import eval_preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr
import os
import argparse

# anchor settings
anchor_ratios = [(1.0, 1.0), (0.4, 1.1), (0.7, 2.5), (1.8, 0.5)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

# category name list
obj_list = ['large-vehicle', 'small-vehicle']

# different resolution for different EfficientDet Families
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

file_dir = os.path.dirname(__file__)


def mkdir(args):
    path = os.path.join(file_dir, args.txt_path)
    if os.path.exists(path):
        print(f'[Info]: {path} has existed.')
    else:
        os.makedirs(path, exist_ok=True)
        print(f'[Info]: {path} being created.')


def get_args():
    parser = argparse.ArgumentParser('Batch inference settings.')
    parser.add_argument('--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--score_threshold', type=float, default=0.6)
    parser.add_argument('--iou_threshold', type=float, default=0.2)
    parser.add_argument('--pth', type=str, default='../logs/rotation_vehicles/efficientdet-d0_48_3200.pth',
                        help='the pth file of trained model.')

    parser.add_argument('--txt_path', type=str, default='result_classname/',
                        help='the output path of batch inference results.')
    parser.add_argument('--img_path', type=str, default='../datasets/rotation_vehicles/val/',
                        help='val set images path')
    parser.add_argument('--file_list', type=str, default='imgnamefile.txt',
                        help='image name list of the val set images.')

    parser.add_argument('--device', type=int, default=0, help='the number of GPU device.')
    args = parser.parse_args()
    return args


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
    path = args.txt_path
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


def batch_inference(args):
    input_size = input_sizes[args.compound_coef]
    model = EfficientDetBackbone(compound_coef=args.compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)

    # load pth file
    model.load_state_dict(torch.load(args.pth, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    if args.use_cuda:
        model = model.cuda(device=args.device)

    path = args.file_list
    imgpath = args.img_path
    content = []
    with open(path, 'r') as f_in:
        lines = f_in.readlines()
        for idx in range(len(lines)):
            line = lines[idx]
            line = line.strip().split(' ')
            content.append(line[0])

    for i in tqdm(range(len(content)), ncols=88):
        filebasename = content[i]
        img_path = os.path.join(imgpath, filebasename + '.jpg')
        try:
            ori_imgs, framed_imgs, framed_metas = eval_preprocess(img_path, max_size=input_size)
        except:
            f'{img_path.split("/")[-1]} is not in {args.img_path}'

        if args.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = Rotation_BBoxTransform()
            clipBoxes = ClipBoxes()
            addBoxes = BBoxAddScores()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes, addBoxes,
                              args.score_threshold, args.iou_threshold)
        out = invert_affine(framed_metas, out)
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


if __name__ == '__main__':
    args = get_args()
    mkdir(args)
    batch_inference(args)
