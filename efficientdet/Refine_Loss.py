# coding=utf-8

# 待优化位置
#
# parent_positive_index = find_index(hor_positive_indices)
#
# anchor_vertex = []
#     for idx in range(len(anchor_widths_pi)):
#
#
# son_positive_index = find_index(rotation_positive_indices)
#


import torch
import torch.nn as nn
import cv2
import numpy as np
import polyiou
import math
from decimal import Decimal
import matplotlib.pyplot as plt

# from efficientdet.utils import BBoxTransform, ClipBoxes
# from utils.utils import postprocess, invert_affine, display


"""分类损失使用Focal Loss损失函数
1.修改回归损失中损失的计算公式
2.修改Smooth L1 损失函数的计算公式 (optional) 
"""


def find_index(lists):
    result_list = []
    for idx in range(len(lists)):
        if lists[idx] == True:
            result_list.append(idx)
    return result_list


def Rectangle_area(rotation_bbox):
    # 适用于rotation_bbox的个数为多个的情况
    # 用来计算旋转框(x, y, w, h, theta)对应的((x1, y1), (x2, y2), (x3, y3), (x4, y4))的4点坐标
    poly_box = []
    for i in range(len(rotation_bbox)):
        single_box = rotation_bbox[i]
        x_c, y_c = single_box[0], single_box[1]
        width, height = single_box[2], single_box[3]
        theta = single_box[4]
        rect = ((x_c, y_c), (width, height), theta)
        poly = np.float32(cv2.boxPoints(rect))  # poly [[x1, y1] [x2, y2] [x3, y3] [x4, y4]]
        # print(f'经过cv2.boxPoints转换后的坐标:\n{poly}')
        poly_box.append(poly)
    return poly_box


def single_Rectangle_area(rotation_bbox):
    # 适用于rotation_bbox列表只有一个的情况
    # 用来计算旋转框(x, y, w, h, theta)对应的((x1, y1), (x2, y2), (x3, y3), (x4, y4))的4点坐标
    x_c, y_c = rotation_bbox[0], rotation_bbox[1]
    width, height = rotation_bbox[2], rotation_bbox[3]
    theta = rotation_bbox[4]
    rect = ((x_c, y_c), (width, height), theta)
    poly = np.float32(cv2.boxPoints(rect))

    return poly


def Rotation2points(Poly):
    # 用于将Rectangle_area()函数的输出转换为
    # [x1, y1, x2, y2, x3, y3, x4, y4]的形式
    lists = []
    for idx in range(len(Poly)):
        lists.append(Poly[idx][0])
        lists.append(Poly[idx][1])
    return lists


def poly2Horizontal_rect(poly):  # poly [[x1, y1] [x2, y2] [x3, y3] [x4, y4]]
    # 由旋转框对应的点坐标转换成最小的内接水平矩形框
    total_list = []

    for idx in range(len(poly)):
        xlist = []
        ylist = []
        for j in range(len(poly[idx])):
            xlist.append(poly[idx][j][0])
            ylist.append(poly[idx][j][1])

        xmin = np.min(xlist)
        xmax = np.max(xlist)
        ymin = np.min(ylist)
        ymax = np.max(ylist)

        total_list.append([xmin, ymin, xmax, ymax])
    return total_list


def calc_iou(a, b):
    # 计算两个水平框的IOU值，用来对旋转框是否相交进行初步的判断
    # Zylo annotation
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]
    # a.shape(49104, 4) b.shape(9,4) a,b --> Tensor
    # debug

    # rotation annotation
    # a(anchor) [x1, y1, x2, y2, theta]
    # b 将旋转的包围框转换成了最小外接水平框(xmin, ymin, xmax, ymax)
    # 目前a, b ---> array  实际运行中, a, b应为Tensor, 为了代码能够运行，先转换成Tensor

    # a = torch.tensor(a).cuda()  # 测试的使用，数组加载到cuda()设备上需要先转换为Tensor张量，然后才能传到cuda上
    # b = torch.tensor(b).cuda()  # 测试的使用，将数组将在到cuda()设备上

    # temp0 = a[:, 3]
    # temp1 = torch.unsqueeze(temp0, dim=1)
    # temp2 = b[:, 2]
    # temp3 = torch.min(temp1, temp2)
    # temp4 = torch.unsqueeze(a[:, 1], 1)
    # temp5 = b[:, 0]
    # temp6 = torch.max(temp4, temp5)
    # temp7 = temp3 - temp6
    #
    #
    # Zylo original code
    # area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    # iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    # ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    # iw = torch.clamp(iw, min=0)
    # ih = torch.clamp(ih, min=0)
    # ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    # ua = torch.clamp(ua, min=1e-8)
    # intersection = iw * ih
    # IoU = intersection / ua

    # Rotation code
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    horizontal_IoU = intersection / ua

    return horizontal_IoU


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        # epsilon = 0.15  # add epsilon for the label smoothing
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        # anchor --> [x1, y1, x2, y2, theta]
        dtype = anchors.dtype

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        anchor_theta = anchor[:, 4]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            # bbox_annotation -->(x_c, y_c, width, height, theta, cls)
            bbox_annotation = bbox_annotation[bbox_annotation[:, 5] != -1]

            # log(prob) if prob is too small, the log(prob) will be Nan
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():

                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:

                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue

            # 正负样本的判别步骤
            # calc_iou() 计算两个水平框的IOU，用来进行初步的判断两个旋转框是否相交

            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()
            # print(f"line 201: {targets}")

            # 1、首先获取gt(bbox_annotation)对应的poly坐标
            # bbox_annotation: (x_c, y_c, width, height, theta, cls), shape[9, 6]

            vertex = Rectangle_area(bbox_annotation[:, :5])  # vertex --> ndarray
            # vertex [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ... [[], [], [], []]]

            # 2、将poly坐标转换成最小的外接水平包围框
            horizontal_vertex = poly2Horizontal_rect(vertex)  # horizontal_vertex --> ndarray
            # horizontal_vertex [[xmin, ymin, xmax, ymax], ... ,[]]

            # 3、计算最小的外接水平包围框与gt(bbox_annotation)的HIoU
            # IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])  # 原始代码

            if torch.cuda.is_available():
                horizontal_vertex = torch.tensor(horizontal_vertex).cuda()

            HIoU = calc_iou(anchor[:, :], horizontal_vertex)  # HIoU.shape(torch.Size([len(anchor), len(anno)])
            hor_IoU_max, hor_IoU_argmax = torch.max(HIoU, dim=1)

            # 根据设定的水平框的IoU阈值(hor_threshold = 0.2)，获得可能使正样本的Anchor index
            hor_positive_indices = torch.ge(hor_IoU_max, 0.4)
            parent_num_list = np.arange(len(anchor))
            parent_positive_index = list(parent_num_list[hor_positive_indices.cpu().numpy()])
            # print(temp_parent_positive_index)
            # parent_positive_index = find_index(hor_positive_indices)  # 原始代码，速度很慢，用parent_num_list[]进行代替
            # print(f"line 188: 经过水平框的初步筛选，水平正类有:{hor_positive_indices.sum()}")

            # 4、将IOU大于设定的hor_threshold的水平包围框对应的旋转框与gt计算skew IOU，然后再计算正类

            # 将所有的Anchor先将gt进行初步分配
            assigned_annotations = bbox_annotation[hor_IoU_argmax, :]

            # 将可能为正类样本的Anchor分配与其匹配的gt信息(x1, y1, width, height, theta, cls)
            # 此时hor_positive_assigned_annotations存放与anchor相匹配的gt信息
            hor_positive_assigned_annotations = assigned_annotations[hor_positive_indices, :]
            # print(f"line 199: 当前每个Anchor对应的annotation:{hor_positive_assigned_annotations.size()}")

            # 获取可能正类样本的Anchor自身坐标及宽高信息用于计算Anchor与gt的skew IoU
            # print(f"line 206: -------->{hor_positive_indices}")

            # 目前hor_positive_indices是tensor，但是anchor_widths_pi等四个变量均是ndarray
            # hor_positive_indices = hor_positive_indices.cpu().numpy()

            anchor_widths_pi = anchor_widths[hor_positive_indices]  # shape(len(hor_positive_indices), )
            anchor_heights_pi = anchor_heights[hor_positive_indices]
            anchor_ctr_x_pi = anchor_ctr_x[hor_positive_indices]
            anchor_ctr_y_pi = anchor_ctr_y[hor_positive_indices]

            # anchor_vertex = []
            # for idx in range(len(anchor_widths_pi)):
            #     xlt, ylt = anchor_ctr_x_pi[idx] - anchor_widths_pi[idx] / 2, anchor_ctr_y_pi[idx] - anchor_heights_pi[idx] / 2
            #     xrt, yrt = anchor_ctr_x_pi[idx] + anchor_widths_pi[idx] / 2, anchor_ctr_y_pi[idx] - anchor_heights_pi[idx] / 2
            #     xrb, yrb = anchor_ctr_x_pi[idx] + anchor_widths_pi[idx] / 2, anchor_ctr_y_pi[idx] + anchor_heights_pi[idx] / 2
            #     xlb, ylb = anchor_ctr_x_pi[idx] - anchor_widths_pi[idx] / 2, anchor_ctr_y_pi[idx] + anchor_heights_pi[idx] / 2
            #
            #     anchor_vertex.append([xlt, ylt, xrt, yrt, xrb, yrb, xlb, ylb])
            # # print(f"line 228: 可能为正样本的Anchor的个数{len(anchor_vertex)}")

            xlt, ylt = anchor_ctr_x_pi - anchor_widths_pi / 2, anchor_ctr_y_pi - anchor_heights_pi / 2
            xrt, yrt = anchor_ctr_x_pi + anchor_widths_pi / 2, anchor_ctr_y_pi - anchor_heights_pi / 2
            xrb, yrb = anchor_ctr_x_pi + anchor_widths_pi / 2, anchor_ctr_y_pi + anchor_heights_pi / 2
            xlb, ylb = anchor_ctr_x_pi - anchor_widths_pi / 2, anchor_ctr_y_pi + anchor_heights_pi / 2

            anchor_vertex = torch.stack([xlt, ylt, xrt, yrt, xrb, yrb, xlb, ylb]).T
            # print(anchor_vertex)
            # 计算可能为正样本的Anchor与gt的 skew IoU  Note:anchor_vertex中anchor[i] 与
            # hor_positive_assigned_annotations[i] 是一一对应的关系
            skew_IoU_lists = []
            for index in range(len(anchor_vertex)):
                single_anchor = anchor_vertex[index]
                single_rotation_opencv_format = hor_positive_assigned_annotations[index][:5]
                single_rotation_box = single_Rectangle_area(single_rotation_opencv_format)
                single_points = Rotation2points(single_rotation_box)

                # example input for polyiou.iou_poly() function
                # temp1 = [433.16364, 186.74037, 421.0, 172.99998, 482.83636, 118.2596, 495.0, 131.99998]
                # temp2 = [282.601, 114.6088, 333.39912, 114.60088, 333.394912, 165.912, 282.6088, 165.912]

                # 由于上面代码中的single_points, single_anchor对应的类型是np.float64类型，函数中需要float类型
                result1 = list(map(lambda x: float(x), single_points))
                result2 = list(map(lambda x: float(x), single_anchor))
                # print(result1)
                # print(result2)
                skew_IoU = polyiou.iou_poly(polyiou.VectorDouble(result1), polyiou.VectorDouble(result2))
                skew_IoU = np.array(skew_IoU).astype(np.float64)
                skew_IoU_lists.append(skew_IoU)

            skew_IoU_lists = np.array(skew_IoU_lists)

            if not torch.is_tensor(skew_IoU_lists):
                overlaps = torch.from_numpy(skew_IoU_lists).cuda(0)
            # print(f"line 285: ----->　{torch.max(overlaps)}")
            rotation_threshold = 0.2  # 使用0.2的阈值可以有较高的检测正类个数
            rotation_positive_indices = torch.ge(overlaps, rotation_threshold)
            # print(f"line 291: ----->  {rotation_positive_indices}")
            # son_positive_index = find_index(rotation_positive_indices)

            son_num_list = np.arange(len(overlaps))
            son_positive_index = list(son_num_list[rotation_positive_indices.cpu().numpy()])  # 对用find_index代码进行优化

            # print(f"经过第二次筛选用于匹配旋转gt的anchor有{len(son_positive_index)}个")

            # target (torch.Tensor): The learning label of the prediction.
            # compute the loss for classification

            # 对正负样本进行one-hot编码
            # 正负样本的划分方式
            # 初次筛选成功的水平框的anchor为负类，其中通过旋转框筛选的anchor为正类

            targets[hor_positive_indices, :] = 0  # ori code
            # targets[torch.lt(IoU_max, 0.4), :] = epsilon  # modified for the label smoothing

            num_positive_anchors = rotation_positive_indices.sum()
            # print(f"正类的个数为{num_positive_anchors}")

            # # 寻找正类样本
            # positive_index_list = []
            # print(f"line 310 :----->{son_positive_index}")
            # for idx in range(len(son_positive_index)):
            #     son_idx = son_positive_index[idx]
            #     parent_idx = parent_positive_index[son_idx]
            #     positive_index_list.append(parent_idx)

            # 对寻找正类样本位置的代码的优化
            # 将positive_index_list 改写为[False, False, True, True, ... False, True]的这种形式
            positive_index_list = np.zeros(len(anchor), dtype=bool)
            for idx in range(len(son_positive_index)):
                son_idx = son_positive_index[idx]
                parent_idx = parent_positive_index[son_idx]
                positive_index_list[parent_idx] = 1

            # for jdx in range(len(positive_index_list)):  # rotation one-hot label
            #     targets[positive_index_list[jdx], :] = 0
            #     targets[positive_index_list[jdx], assigned_annotations[positive_index_list[jdx], 5].long()] = 1

            # 对上面两行代码的优化
            targets[positive_index_list, :] = 0
            targets[positive_index_list, assigned_annotations[positive_index_list, 5].long()] = 1

            # Zylo code
            # targets[rotation_positive_indices, :] = 0
            # targets[rotation_positive_indices, assigned_annotations[rotation_positive_indices, 5].long()] = 1

            # label smoothing
            # targets[positive_indices, :] = epsilon  # modified for the label smoothing
            # targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1.0 - epsilon

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            # Zylo code
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)

            # label smoothing
            # alpha_factor = torch.where(torch.eq(targets, 1.0 - epsilon), alpha_factor, 1. - alpha_factor)
            # focal_weight = torch.where(torch.eq(targets, 1.0 - epsilon), 1. - classification, classification)

            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
            # print(f"分类损失：{classification_losses}")

            # rotation regression loss begin
            if rotation_positive_indices.sum() > 0:
                rotation_assigned_annotations_gt = hor_positive_assigned_annotations[rotation_positive_indices]
                # print(f"line 356: -----> {rotation_assigned_annotations_gt}")

                # anchor -> width, height, center_x, center_y
                # rotation_anchor_widths_pi = []
                # rotation_anchor_heights_pi = []
                # rotation_anchor_ctr_x_pi = []
                # rotation_anchor_ctr_y_pi = []
                # rotation_anchor_theta = []
                #
                # for idx in range(len(positive_index_list)):
                #     rotation_anchor_widths_pi.append(anchor_widths[positive_index_list[idx]])
                #     rotation_anchor_heights_pi.append(anchor_heights[positive_index_list[idx]])
                #     rotation_anchor_ctr_x_pi.append(anchor_ctr_x[positive_index_list[idx]])
                #     rotation_anchor_ctr_y_pi.append(anchor_ctr_y[positive_index_list[idx]])
                #     rotation_anchor_theta.append(anchor_theta[positive_index_list[idx]])

                rotation_anchor_widths_pi = anchor_widths[positive_index_list]
                rotation_anchor_heights_pi = anchor_heights[positive_index_list]
                rotation_anchor_ctr_x_pi = anchor_ctr_x[positive_index_list]
                rotation_anchor_ctr_y_pi = anchor_ctr_y[positive_index_list]
                rotation_anchor_theta = anchor_theta[positive_index_list]


                    # rotation_gt_widths.append(assigned_annotations[positive_index_list[idx]][2])
                    # rotation_gt_heights.append(assigned_annotations[positive_index_list[idx]][3])
                    # rotation_gt_ctr_x.append(assigned_annotations[positive_index_list[idx]][0])
                    # rotation_gt_ctr_y.append(assigned_annotations[positive_index_list[idx]][1])
                    # rotation_gt_theta.append(assigned_annotations[positive_index_list[idx]][4])

                # Zylo code
                # # GT -> width, height, center_x, center_y
                # gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                # gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                # gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                # gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                regression_loss = []
                for ind in range(len(rotation_anchor_theta)):
                    single_gt_width = torch.clamp(rotation_assigned_annotations_gt[ind][2], min=1)
                    single_gt_height = torch.clamp(rotation_assigned_annotations_gt[ind][3], min=1)

                    targets_dx = (rotation_assigned_annotations_gt[ind][0] - rotation_anchor_ctr_x_pi[ind]) / \
                                 rotation_anchor_widths_pi[ind]
                    targets_dy = (rotation_assigned_annotations_gt[ind][1] - rotation_anchor_ctr_y_pi[ind]) / \
                                 rotation_anchor_heights_pi[ind]
                    targets_dw = torch.log(single_gt_width / rotation_anchor_widths_pi[ind])
                    targets_dh = torch.log(single_gt_height / rotation_anchor_heights_pi[ind])
                    targets_theta = (rotation_assigned_annotations_gt[ind][4] / 180 * math.pi) - (
                                (rotation_anchor_theta[ind] / 180) * math.pi)

                    # # tx, ty, tw, th -> the transform between anchors and GTs
                    # targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    # targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                    # targets_dw = torch.log(gt_widths / anchor_widths_pi)
                    # targets_dh = torch.log(gt_heights / anchor_heights_pi)

                    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw, targets_theta))

                    single_predict = regression[parent_positive_index[son_positive_index[ind]], :]

                    single_regression_diff = torch.abs(targets - single_predict)

                    single_regression_loss = torch.where(
                        torch.le(single_regression_diff, 1.0 / 9.0),
                        0.5 * 9.0 * torch.pow(single_regression_diff, 2),
                        single_regression_diff - 0.5 / 9.0
                    )
                regression_losses.append(single_regression_loss.mean())
                # print(f"回归损失：{regression_losses}")
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        # imgs = kwargs.get('imgs', None)
        # if imgs is not None:
        #     regressBoxes = BBoxTransform()
        #     clipBoxes = ClipBoxes()
        #     obj_list = kwargs.get('obj_list', None)
        #     out = postprocess(imgs.detach(),
        #                       torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
        #                       regressBoxes, clipBoxes,
        #                       0.5, 0.3)
        #     imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
        #     imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
        #     imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
        #     display(out, imgs, obj_list, imshow=False, imwrite=True)
        #
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0,
                                                   keepdim=True) * 50  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233


if __name__ == '__main__':
    """line 266 - 338用来测试函数及绘制初筛Anchor思路的图片

    # def draw_line(lists, color='r'):
    #     # lists = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # 
    #     line1x = [lists[0][0], lists[1][0]]
    #     line1y = [lists[0][1], lists[1][1]]
    # 
    #     line2x = [lists[1][0], lists[2][0]]
    #     line2y = [lists[1][1], lists[2][1]]
    # 
    #     line3x = [lists[2][0], lists[3][0]]
    #     line3y = [lists[2][1], lists[3][1]]
    # 
    #     line4x = [lists[3][0], lists[0][0]]
    #     line4y = [lists[3][1], lists[0][1]]
    # 
    #     plt.plot(line1x, line1y, c=color)
    #     plt.plot(line2x, line2y, c=color)
    #     plt.plot(line3x, line3y, c=color)
    #     plt.plot(line4x, line4y, c=color)
    # 
    # 
    # def draw_point(lists, color='r'):
    #     # [[x1, x2, x3, x4], [y1, y2, y3, y4]]
    #     xlist = lists[0]
    #     ylist = lists[1]
    # 
    #     plt.scatter(xlist, ylist, c=color)
    # 
    # 
    # rotation_bbox = [[3.0, 3.0, 2.828, 1.414, -45.0]]
    # horizontal_bbox, vertex = Rectangle_area(rotation_bbox)  # vertex:[[x1, y1] [x2, y2] [x3, y3] [x4, y4]]
    # min_vertex = poly2rect(vertex)
    # 
    # # 为了画图需要
    # xmin, ymin = min_vertex[0][0], min_vertex[0][1]
    # xmax, ymax = min_vertex[1][0], min_vertex[1][0]
    # minarea_point = [[xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]]
    # draw_point(minarea_point, color='g')
    # minarea_line = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    # draw_line(minarea_line, color='g')
    # 
    # x = []
    # y = []
    # rotation_line = []
    # for i in range(len(vertex)):
    #     x.append(vertex[i][0])
    #     y.append(vertex[i][1])
    #     rotation_line.append([vertex[i][0], vertex[i][1]])
    # draw_line(rotation_line, color='r')
    # 
    # # anchor
    # x1 = [2.0, 5, 5, 2.0]
    # y1 = [2.0, 2.0, 5, 5]
    # anchor_line = [[2.0, 2.0], [5, 2.0], [5, 5], [2.0, 5]]
    # draw_line(anchor_line, color='b')
    # anchor_point = []
    # anchor_point.append(x1)
    # anchor_point.append(y1)
    # draw_point(anchor_point, color='b')
    # 
    # rotation_point = []
    # rotation_point.append(x)
    # rotation_point.append(y)
    # draw_point(rotation_point, color='r')
    # 
    # plt.plot()
    # ax = plt.gca()
    # ax.set_xlim(left=0, right=10)
    # ax.set_ylim(bottom=10, top=0)  # 此处将原点设置为左上角
    # ax.xaxis.tick_top()
    # plt.show() """

    """ 下面的代码用来测试相应的初筛函数等内容 """
    from torchvision import transforms
    import os
    from torch.utils.data import DataLoader
    import argparse
    import yaml

    from efficientdet.utils import Anchors

    from efficientdet.rotation_dataset import RotationCocoDataset, collater
    from efficientdet.rotation_dataset import Normalizer, Resizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default=r'/home/fzh/Rotation-EfficinetDet/datasets/')
    args = parser.parse_args()

    yaml_rootpath = r'/home/fzh/Rotation-EfficinetDet/projects/'
    yamlpath = os.path.join(yaml_rootpath, 'rotation_vehicles.yml')


    class Params:
        def __init__(self, project_file):
            self.params = yaml.safe_load(open(project_file).read())

        def __getattr__(self, item):
            return self.params.get(item, None)


    params = Params(yamlpath)
    test_batch_size = 8
    training_params = {'batch_size': test_batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': 12}

    training_set = RotationCocoDataset(
        root_dir=os.path.join(args.root_path, params.project_name),
        set=params.train_set,
        transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                      Resizer(512)]))

    training_generator = DataLoader(training_set, **training_params)
    num_iter = len(training_generator)  # num_iter = 135 len(training set) = 135 * 8 = 1080

    dataiter = iter(training_generator)
    iter_content = dataiter.next()
    #
    # iter_content 字典类型，即每次送入到网络模型中的数据，该字典包含三个key: 'img', 'annot', 'scale'
    # 'img'：包含了送入到网络中的图片 shape(torch.size([batch_size, 3, 512, 512]))
    # 'annot'：包含gt标注(x_c, y_c, width, height, theta, cls)

    # load the annotation of the batch_size

    # 获取分析所需要的anno(gt)信息，用于进行损失的计算

    # 将annot加载到CUDA上
    annot = iter_content['annot']
    annotation = annot.cuda(0)

    # 将img加载到CUDA上
    img = iter_content['img']
    images = img.cuda(0)

    anchor = Anchors()
    anchors = anchor.forward(images)  # 利用dataset.Anchor()产生的anchor正确，且加载到了CUDA上

    # 计算损失
    FL = FocalLoss()
    classifications = torch.from_numpy(np.ones([test_batch_size, 49104, 2]) * 0.5).cuda(0)
    regressions = torch.from_numpy(np.ones([test_batch_size, 49104, 5]) * 0.5).cuda(0)
    FL.forward(classifications=classifications, regressions=regressions, anchors=anchors, annotations=annotation)



