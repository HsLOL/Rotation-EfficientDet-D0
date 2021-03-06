"""The script is used to calculate the rotation loss, modified from original code."""
# coding=utf-8

import torch
import torch.nn as nn
import cv2
import numpy as np
import math
from random import sample
from polyiou import polyiou


def find_index(lists):
    result_list = []
    for idx in range(len(lists)):
        if lists[idx] == True:
            result_list.append(idx)
    return result_list


def visualize_Rectangle_area(image_path, lists):
    image = cv2.imread(image_path)
    rect = np.array(np.int0(lists))
    cv2.drawContours(image=image,
                     contours=rect,
                     contourIdx=-1,
                     color=[0, 0, 255],
                     thickness=2)

    cv2.imwrite('{Output_path}/visualize_Rectangle_area.jpg',
                image)


def visualize_poly2Horizontal_rect(image_path, lists):
    # 类似visualize_positive_anchor，给定左上角和右下角绘制矩形框
    img = cv2.imread(image_path)
    for idx in range(len(lists)):
        lists[idx][:4] = map(int, lists[idx][:4])
        xmin, ymin, xmax, ymax = lists[idx][0], lists[idx][1], lists[idx][2], lists[idx][3]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=[0, 255, 0], thickness=1)
    cv2.imwrite('{Output_path}/visualize_visualize_poly2Horizontal_rect.jpg',
                img)


def visualize_positive_anchor(image_path, anchor_lists, index_lists):
    # 根据左上角和右下角的点绘制矩形框
    anchor_lists = anchor_lists.cpu().numpy().tolist()
    image = cv2.imread(image_path)

    sample_list = sample(index_lists, len(index_lists))
    for idx in sample_list:
        single_anchor = anchor_lists[idx]

        single_anchor[:4] = map(int, single_anchor[:4])
        x1, y1, x2, y2 = single_anchor[0], single_anchor[1], single_anchor[2], single_anchor[3]
        cv2.rectangle(image, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=1)

    cv2.imwrite('{Output_path}/visualize_positive_anchor.jpg',
                image)


def visualize_rp_anchor(path, lists):
    image = cv2.imread(path)
    if torch.is_tensor(lists):
        lists = lists.cpu().numpy().tolist()

    for idx in range(len(lists)):
        lists[idx][:4] = map(int, lists[idx][:4])
        xlt, ylt, xrb, yrb = lists[idx][0], lists[idx][1], lists[idx][2], lists[idx][3]

        cv2.rectangle(image, (xlt, ylt), (xrb, yrb), color=[255, 0, 0], thickness=1)
    cv2.imwrite('{Output_path}/visualize_rp_anchor.jpg', image)


def check_anchor(path, lists):
    image = cv2.imread(path)
    lists = lists.cpu().numpy().tolist()
    for idx in range(len(lists)):
        single_list = lists[idx]
        single_list[:] = map(int, single_list[:])
        xmin, ymin, xmax, ymax = single_list[0], single_list[1], single_list[4], single_list[5]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=[255, 255, 0], thickness=1)
    cv2.imwrite('{Output_path}/check_anchor.jpg', image)


def visualize_pp_anchor(path, lists):
    image = cv2.imread(path)
    if torch.is_tensor(lists):
        lists = lists.cpu().numpy().tolist()

    for idx in range(len(lists)):
        templist = list(map(int, lists[idx][:4]))
        xmin, ymin, xmax, ymax = templist[0], templist[1], templist[2], templist[3]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=[0, 240, 0], thickness=1)
    cv2.imwrite('{Output_path}/visualize_pp_anchor.jpg', image)


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
    # rotation annotation
    # a(anchor) [x1, y1, x2, y2, theta]
    # b 将旋转的包围框转换成了最小外接水平框(xmin, ymin, xmax, ymax)
    # 目前a, b ---> array  实际运行中, a, b应为Tensor, 为了代码能够运行，先转换成Tensor

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
        # epsilon = 0.15  # used for label smoothing

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # shape:[xmin, ymin, xmax, ymax, theta]
        dtype = anchors.dtype

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        anchor_theta = anchor[:, 4]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]  # bbox_annotation -->(x_c, y_c, width, height, theta, cls)
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

            # Get positive and negative samples
            # Step1. Get horizontal overlaps between anchors and gt box

            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            # 1) 首先获取gt(bbox_annotation)对应的poly坐标
            vertex = Rectangle_area(bbox_annotation[:, :5])  # vertex --> ndarray

            # Debug
            # visualize_Rectangle_area(image_path, vertex[:5])
            # vertex [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ... [[], [], [], []]]

            # 2) 将poly坐标转换成最小的外接水平包围框
            horizontal_vertex = poly2Horizontal_rect(vertex)  # horizontal_vertex --> ndarray

            # Debug
            # visualize_poly2Horizontal_rect(image_path, horizontal_vertex[:5])
            # horizontal_vertex [[xmin, ymin, xmax, ymax], ... ,[]]

            # 3) 计算最小的外接水平包围框与gt(bbox_annotation)的horizontal overlaps
            if torch.cuda.is_available():
                horizontal_vertex = torch.tensor(horizontal_vertex).cuda()

            HIoU = calc_iou(anchor[:, :], horizontal_vertex)  # HIoU shape (torch.Size([len(anchor), len(anno)])
            hor_IoU_max, hor_IoU_argmax = torch.max(HIoU, dim=1)

            # Step2. 根据设定的水平框的IoU阈值(hor_threshold = 0.2)，获得可能使正样本的Anchor index
            hor_positive_indices = torch.ge(hor_IoU_max, 0.6)
            # print(f"第一次水平框筛选后：水平正类的个数为{hor_positive_indices.sum()}")
            parent_num_list = np.arange(len(anchor))
            parent_positive_index = list(parent_num_list[hor_positive_indices.cpu().numpy()])

            positive_anchor_list = anchor[hor_positive_indices, :]

            # Debug
            # visualize_positive_anchor(image_path, anchor, parent_positive_index)

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

            xlt, ylt = anchor_ctr_x_pi - anchor_widths_pi / 2, anchor_ctr_y_pi - anchor_heights_pi / 2
            xrt, yrt = anchor_ctr_x_pi + anchor_widths_pi / 2, anchor_ctr_y_pi - anchor_heights_pi / 2
            xrb, yrb = anchor_ctr_x_pi + anchor_widths_pi / 2, anchor_ctr_y_pi + anchor_heights_pi / 2
            xlb, ylb = anchor_ctr_x_pi - anchor_widths_pi / 2, anchor_ctr_y_pi + anchor_heights_pi / 2

            anchor_vertex = torch.stack([xlt, ylt, xrt, yrt, xrb, yrb, xlb, ylb]).t()
            # 可视化anchor_vertex表示的是否正确
            # check_anchor(image_path, anchor_vertex)

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
                skew_IoU = polyiou.iou_poly(polyiou.VectorDouble(result1), polyiou.VectorDouble(result2))
                skew_IoU = np.array(skew_IoU).astype(np.float64)
                skew_IoU_lists.append(skew_IoU)

            skew_IoU_lists = np.array(skew_IoU_lists)

            if not torch.is_tensor(skew_IoU_lists):
                overlaps = torch.from_numpy(skew_IoU_lists).cuda(0)
            # print(f"line 285: ----->　{torch.max(overlaps)}")
            rotation_threshold = 0.3  # 使用0.2的阈值可以有较高的检测正类个数
            rotation_positive_indices = torch.ge(overlaps, rotation_threshold)
            # print(f"line 291: ----->  {rotation_positive_indices}")
            # son_positive_index = find_index(rotation_positive_indices)

            son_num_list = np.arange(len(overlaps))
            son_positive_index = list(son_num_list[rotation_positive_indices.cpu().numpy()])  # 对用find_index代码进行优化

            # 可视化此时正类Anchor的筛选情况
            # print(f"第二次筛选后，Anchor的数量情况{len(son_positive_index)}")
            # positive_index_list = np.zeros(len(anchor), dtype=bool)
            # for idx in range(len(son_positive_index)):
            #     son_idx = son_positive_index[idx]
            #     parent_idx = parent_positive_index[son_idx]
            #     positive_index_list[parent_idx] = 1
            # pp_anchor = anchor[positive_index_list]
            #
            # visualize_pp_anchor(image_path, pp_anchor)

            # target (torch.Tensor): The learning label of the prediction.
            # compute the loss for classification

            # 对正负样本进行one-hot编码
            # 正负样本的划分方式 Zylo117 pos->169 neg->48558
            # 初次筛选成功的水平框的anchor为负类，其中通过旋转框筛选的anchor为正类

            # targets[hor_positive_indices, :] = 0  # ori code
            # targets[torch.lt(IoU_max, 0.4), :] = epsilon  # modified for the label smoothing
            # temp = torch.lt(hor_IoU_max, 0.4)
            # neg_num = temp.sum()

            # hor_IoU_max < 0.4 设置为负类样本
            targets[torch.lt(hor_IoU_max, 0.4), :] = 0

            # skew_IoU > 0.3 设置为正类样本
            num_positive_anchors = rotation_positive_indices.sum()

            # # 寻找正类样本
            # 将positive_index_list 改写为[False, False, True, True, ... False, True]的这种形式
            positive_index_list = np.zeros(len(anchor), dtype=bool)
            for idx in range(len(son_positive_index)):
                son_idx = son_positive_index[idx]
                parent_idx = parent_positive_index[son_idx]
                positive_index_list[parent_idx] = 1

            targets[positive_index_list, :] = 0
            targets[positive_index_list, assigned_annotations[positive_index_list, 5].long()] = 1

            # for label smoothing
            # targets[positive_indices, :] = epsilon  # modified for the label smoothing
            # targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1.0 - epsilon

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)

            # for label smoothing
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

            # rotation regression loss begin
            if rotation_positive_indices.sum() > 0:

                rotation_assigned_annotations_gt = hor_positive_assigned_annotations[rotation_positive_indices]

                rotation_anchor_widths_pi = anchor_widths[positive_index_list]
                rotation_anchor_heights_pi = anchor_heights[positive_index_list]
                rotation_anchor_ctr_x_pi = anchor_ctr_x[positive_index_list]
                rotation_anchor_ctr_y_pi = anchor_ctr_y[positive_index_list]
                rotation_anchor_theta = anchor_theta[positive_index_list]

                # efficientdet style
                rotation_assigned_annotations_gt[:, 2] = torch.clamp(rotation_assigned_annotations_gt[:, 2], min=1)
                rotation_assigned_annotations_gt[:, 3] = torch.clamp(rotation_assigned_annotations_gt[:, 3], min=1)

                targets_dx = (rotation_assigned_annotations_gt[:, 0] - rotation_anchor_ctr_x_pi) / rotation_anchor_widths_pi
                targets_dy = (rotation_assigned_annotations_gt[:, 1] - rotation_anchor_ctr_y_pi) / rotation_anchor_heights_pi
                targets_dw = torch.log(rotation_assigned_annotations_gt[:, 2] / rotation_anchor_widths_pi)
                targets_dh = torch.log(rotation_assigned_annotations_gt[:, 3] / rotation_anchor_heights_pi)
                targets_theta = ((rotation_assigned_annotations_gt[:, 4] / 180 * math.pi) - (rotation_anchor_theta / 180 * math.pi))

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_theta))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_index_list, :])

                # smooth l1 loss with beta
                # regression_loss = torch.where(
                #     torch.le(regression_diff, 1.0 / 9.0),
                #     0.5 * 9.0 * torch.pow(regression_diff, 2),
                #     regression_diff - 0.5 / 9.0
                # )

                # smooth l1 loss
                regression_loss = torch.where(
                    torch.le(regression_diff, 1),
                    0.5 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5

                )

                regression_losses.append(regression_loss.mean())

            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0,
                                                   keepdim=True) * 50
