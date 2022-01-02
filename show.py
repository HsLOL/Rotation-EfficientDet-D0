"""Visualize the inference result of the trained model."""
import argparse
import torch
import matplotlib.pyplot as plt
from backbone import EfficientDetBackbone
import cv2
import numpy as np
from efficientdet.rotation_utils import Rotation_BBoxTransform, ClipBoxes, BBoxAddScores
from utils.rotation_utils import eval_preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr

# anchor settings with k-means results
anchor_ratios = [(1.0, 1.0), (0.4, 1.1), (0.7, 2.5), (1.8, 0.5)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

# Different resolution for different EfficientDet families
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

# category name list
obj_list = ['large-vehicle', 'small-vehicle']
color_list = standard_to_bgr(STANDARD_COLORS)


def get_args():
    parser = argparse.ArgumentParser('Show the inference result of trained model.')
    parser.add_argument('--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('--img_path', type=str, default='./test/demo2.jpg', help='the path of the inference image')
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--score_threshold', type=float, default=0.6)
    parser.add_argument('--iou_threshold', type=float, default=0.2)
    parser.add_argument('--pth', type=str, default='./logs/rotation_vehicles/efficientdet-d0_48_3200.pth',
                        help='the pth file of trained model.')
    parser.add_argument('--output_path', type=str, default='./test', help='the output path of the inference image.')
    parser.add_argument('--device', type=int, default=0, help='the number of GPU device.')
    args = parser.parse_args()
    return args


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
            """preds[i]['rois'][j] = [xmin, ymin, xmax, ymax, theta]"""

            xmin, ymin, xmax, ymax, theta = preds[i]['rois'][j].astype(np.float)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            color = [0, 0, 255]

            # rotation detection code
            rect = OPENCV2xywh([xmin, ymin, xmax, ymax, theta])
            rect = np.int0(rect)
            rect = np.array(rect)
            cv2.drawContours(
                image=imgs[i],
                contours=rect,
                contourIdx=-1,
                color=color,
                thickness=2)

        if imshow:
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
            plt.imshow(imgs[i])
            plt.show()

        if imwrite:
            name = (args.img_path.split('/')[-1]).split('.')[0]
            cv2.imwrite(f'{args.output_path}/{name}_detected.jpg', imgs[i])


def show(args):
    input_size = input_sizes[args.compound_coef]
    ori_imgs, framed_imgs, framed_metas = eval_preprocess(args.img_path, max_size=input_size)
    if args.use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=args.compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)

    model.load_state_dict(torch.load(args.pth, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()
    if args.use_cuda:
        model = model.cuda(device=args.device)

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
        display(out, ori_imgs, imshow=True, imwrite=False)


if __name__ == '__main__':
    args = get_args()
    show(args)
