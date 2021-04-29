# coding=utf-8

"""Todo
1. Finished the multiple images
2. Visualize the transform images and annotations
"""


# This is the file tree for this code
# root_path --|
#           --|images --| *.png
#
#           --|labelTxt  --| *.txt Note:包含 Google Earth 和 gsd 两行信息 !!!
#
#
# out_root_path --|
#               --|Aug_t_path --|images --|*.png
#                             --|labelTxt --|*.txt
#
#
#               --|Aug_h_path --|images --|*.png
#                             --|labelTxt --|*.txt
#
#               --|Aug_v_path --|images --|*.png
#                             --|labelTxt --|*.txt
#

import os
import cv2
import numpy as np
import math
import random
from collections import defaultdict
import argparse
from .DOTA2opencv import check_theta_and_modify
from .DOTA2opencv import print_list, OPENCV2xywh


def VisualizeOPENCVformat(root_path, visualize_name):
    # root_path 读取图片和anno的根目录, visualize_name是想要可视化图片的名称
    # 将OPENCV格式转换后的anno读取出来，进行可视化检查
    img = cv2.imread(os.path.join(root_path, 'images', visualize_name + '.png'))
    with open(os.path.join(root_path, 'labelTxt', visualize_name + '.txt'), 'r') as f_in:
        lines = f_in.readlines()
        head = lines[:2]
        body = lines[2:]
        content = []
        for line in body:
            line = line.strip().split(' ')
            content.append(line)
        # check theta
        print(f'---- 检查经过编写函数转换后{visualize_name}的OPENCV格式是否含有角度问题 ----')
        modified_list, zero_idx, positive_idx, vertical_idx = check_theta_and_modify(content)
        # print_list(modified_list)

        if len(zero_idx) + len(positive_idx) == 0:
            print('已经不存在角度问题')
        else:
            temp_zero_list = []
            for idx in zero_idx:
                temp_zero_list.append(content[idx])
            print(f'获取的theta=0的标注格式: {print_list(temp_zero_list)}')

        rect = OPENCV2xywh(modified_list)
        rect = np.int0(rect)
        rect = np.array(rect)
        print(f'line66:{rect}')

        cv2.drawContours(image=img,
                         contours=rect,
                         contourIdx=-1,
                         color=[0, 0, 255],
                         thickness=2)
        cv2.imwrite('Draw_Opencv{}.jpg'.format(visualize_name), img)


def write_img_label_Augment(args, lists):
    head = ['imagessource:GoogleEarth', 'gsd:0.120384949364']  # for general
    format = ['vertical', 'horizontal', 'rotation']
    for idx in range(len(lists)):
        trans_foramt = format[idx]
        format_dict = lists[idx]
        # print(f'line21:{format_dict}')
        write_line_list = format_dict['poly']
        # print(f'line24: {check_theta_and_modify(write_line_list)}')
        new_txt_name = format_dict['base_name'] + '.txt'
        new_img_name = format_dict['base_name'] + '.png'
        img = format_dict['trans_img']
        if trans_foramt == 'vertical':
            out_root_path = args.Aug_v_path
        if trans_foramt == 'horizontal':
            out_root_path =args.Aug_h_path
        if trans_foramt == 'rotation':
            out_root_path = args.Aug_t_path

        cv2.imwrite(os.path.join(args.out_root_path, out_root_path, 'images', new_img_name), img)
        with open(os.path.join(args.out_root_path, out_root_path, 'labelTxt', new_txt_name), 'w') as f_out:
            for j in range(len(head)):
                f_out.write(head[j] + '\n')

            for i in range(len(write_line_list)):
                line_content = write_line_list[i]
                out_line = str(line_content[0]) + ' ' + str(line_content[1]) + ' ' + str(line_content[2]) \
                           + ' ' + str(line_content[3]) + ' ' + str(line_content[4]) + ' ' + line_content[5] + ' ' + \
                           line_content[6]
                f_out.write(out_line + '\n')


def make_dirs(paths):
    os.makedirs(os.path.join(paths, 'images'), exist_ok=True)
    os.makedirs(os.path.join(paths, 'labelTxt'), exist_ok=True)


def to_make_dirs(args):
    Aug_h_path_root = os.path.join(args.out_root_path, args.Aug_h_path)
    Aug_v_path_root = os.path.join(args.out_root_path, args.Aug_v_path)
    Aug_t_path_root = os.path.join(args.out_root_path, args.Aug_t_path)

    make_dirs(Aug_h_path_root)
    make_dirs(Aug_v_path_root)
    make_dirs(Aug_t_path_root)


def Rotate_vertex(poly, Pi_angle, a, b):
    # 旋转anno的坐标
    X0 = (poly[0][0] - a) * math.cos(Pi_angle) - (poly[0][1] - b) * math.sin(Pi_angle) + a
    Y0 = (poly[0][0] - a) * math.sin(Pi_angle) + (poly[0][1] - b) * math.cos(Pi_angle) + b

    X1 = (poly[1][0] - a) * math.cos(Pi_angle) - (poly[1][1] - b) * math.sin(Pi_angle) + a
    Y1 = (poly[1][0] - a) * math.sin(Pi_angle) + (poly[1][1] - b) * math.cos(Pi_angle) + b

    X2 = (poly[2][0] - a) * math.cos(Pi_angle) - (poly[2][1] - b) * math.sin(Pi_angle) + a
    Y2 = (poly[2][0] - a) * math.sin(Pi_angle) + (poly[2][1] - b) * math.cos(Pi_angle) + b

    X3 = (poly[3][0] - a) * math.cos(Pi_angle) - (poly[3][1] - b) * math.sin(Pi_angle) + a
    Y3 = (poly[3][0] - a) * math.sin(Pi_angle) + (poly[3][1] - b) * math.cos(Pi_angle) + b

    return np.array([(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)])


def Argument(args, ori_lists, is_horizontal=None, is_vertical=None, is_rotation=None):
    for idx in range(len(ori_lists)):
        single_dict = ori_lists[idx]  # a single_dict is a pic
        # print(single_dict)
        base_name = single_dict['basename']
        img = cv2.imread(os.path.join(args.root_path, 'images', base_name + '.png'))

        width, height = single_dict['shape']['width'], single_dict['shape']['height']  # width, height of image
        horizontal_list = []
        if is_horizontal:
            dict_horizontal = {}
            img_h = cv2.flip(img, 1)
            for i in range(len(single_dict['poly'])):
                single_obj = single_dict['poly'][i]
                x_c, y_c = float(single_obj[0]), float(single_obj[1])
                obj_w, obj_h = float(single_obj[2]), float(single_obj[3])
                obj_theta = float(single_obj[4])
                cls = single_obj[5]
                diff = single_obj[6]

                h_x_c = width - x_c
                h_y_c = y_c
                if obj_theta == -90:
                    h_theta = obj_theta
                    h_obj_h = obj_h
                    h_obj_w = obj_w
                else:
                    h_theta = float((90 - abs(obj_theta)) * -1)
                    h_obj_w = obj_h
                    h_obj_h = obj_w
                horizontal_list.append([h_x_c, h_y_c, h_obj_w, h_obj_h, h_theta, cls, diff])
        dict_horizontal['base_name'] = base_name
        dict_horizontal['poly'] = horizontal_list
        dict_horizontal['trans_img'] = img_h

        vertical_list = []
        if is_vertical:
            dict_vertical = {}
            img_v = cv2.flip(img, 0)
            for i in range(len(single_dict['poly'])):
                single_obj = single_dict['poly'][i]
                x_c, y_c = float(single_obj[0]), float(single_obj[1])
                obj_w, obj_h = float(single_obj[2]), float(single_obj[3])
                obj_theta = float(single_obj[4])
                cls = single_obj[5]
                diff = single_obj[6]

                v_x_c, v_y_c = x_c, height - y_c

                if obj_theta == -90:
                    v_theta = obj_theta
                    v_obj_w = obj_w
                    v_obj_h = obj_h
                else:
                    v_theta = float((90 - abs(obj_theta)) * -1)
                    v_obj_w = obj_h
                    v_obj_h = obj_w
                vertical_list.append([v_x_c, v_y_c, v_obj_w, v_obj_h, v_theta, cls, diff])
        dict_vertical['base_name'] = base_name
        dict_vertical['poly'] = vertical_list
        dict_vertical['trans_img'] = img_v

        theta_list = []
        if is_rotation:
            dict_rotation = {}
            base_degree = 45
            rotate_angle = random.uniform(-base_degree, base_degree) + 10
            print(f'line208:{rotate_angle}')
            radin_angle = -rotate_angle * math.pi / 180.0
            rotation_center_x, rotation_center_y = width / 2, height / 2
            M = cv2.getRotationMatrix2D(center=(rotation_center_x, rotation_center_y), angle=rotate_angle, scale=1)
            rotated_img = cv2.warpAffine(img, M, (width, height))

            for i in range(len(single_dict['poly'])):
                single_obj = single_dict['poly'][i]
                x_c, y_c = float(single_obj[0]), float(single_obj[1])
                obj_w, obj_h = float(single_obj[2]), float(single_obj[3])
                obj_theta = float(single_obj[4])
                cls = single_obj[5]
                diff = single_obj[6]

                # rect = [(x_c, y_c), (w, h), theta]
                rect = ((x_c, y_c), (obj_w, obj_h), obj_theta)
                # print(f'line223{rect}')
                vertex = cv2.boxPoints(rect)  # vertex = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                vertex_arr = Rotate_vertex(vertex, radin_angle, rotation_center_x, rotation_center_y)
                rect_rotated = cv2.minAreaRect(np.float32(vertex_arr))

                rotation_c_x = rect_rotated[0][0]
                rotation_c_y = rect_rotated[0][1]
                rotation_w = rect_rotated[1][0]
                rotation_y = rect_rotated[1][1]
                rotation_theta = rect_rotated[-1]
                theta_list.append([rotation_c_x, rotation_c_y, rotation_w, rotation_y, rotation_theta, cls, diff])
            dict_rotation['base_name'] = base_name
            dict_rotation['poly'] = theta_list
            dict_rotation['trans_img'] = rotated_img

        return dict_vertical, dict_horizontal, dict_rotation


def load_anno_and_make_new_anno(args):
    txt_path = os.path.join(args.root_path, 'labelTxt')
    img_path = os.path.join(args.root_path, 'images')

    txt_list = os.listdir(txt_path)
    # print(txt_list)
    anno_list = []

    for idx in range(len(txt_list)):
        whole_dict = {}
        shape_dict = {}
        basename = os.path.splitext(os.path.basename(txt_list[idx]))[0]
        img_name = basename + '.png'
        txt_name = basename + '.txt'

        img = cv2.imread(os.path.join(img_path, img_name))
        height, width, _ = img.shape
        shape_dict['width'] = width
        shape_dict['height'] = height
        whole_dict['basename'] = basename
        whole_dict['shape'] = shape_dict

        with open(os.path.join(txt_path, txt_name)) as f_in:
            lines = f_in.readlines()
            head = lines[:2]
            body = lines[2:]
            file_content = []
            for line in body:
                line = line.strip().split(' ')
                file_content.append(line)
            whole_dict['poly'] = file_content  # anno_dict {'P0005': [{'width': 977, 'height': 884}, {'poly': [['76.0', '847.0', '76.0', '864.0', '34.0',....]]}]
        anno_list.append(whole_dict)
    return anno_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default=r'/home/fzh/DOTA_devkit_YOLO-master/rotation_test/')

    parser.add_argument('--Aug_h_path', type=str,
                        default=r'Augment_horizontal/')

    parser.add_argument('--Aug_v_path', type=str,
                        default=r'Augment_vertical/')

    parser.add_argument('--Aug_t_path', type=str,
                        default=r'Augment_rotation/')

    parser.add_argument('--out_root_path', type=str,
                        default=r'./Data_Augment/')

    args = parser.parse_args()

    to_make_dirs(args)

    anno_lists = load_anno_and_make_new_anno(args)

    vertical_dict, horizontal_dict, rotational_dict = Argument(args, anno_lists, is_horizontal=True, is_vertical=True, is_rotation=True)

    write_img_label_Augment(args, [vertical_dict, horizontal_dict, rotational_dict])

    # 可视化水平翻转、垂直翻转、旋转的结果
    VisualizeOPENCVformat(os.path.join(args.out_root_path, args.Aug_t_path), 'P0005')

    # 可视化原图
    # VisualizeOPENCVformat(args.root_path, 'P0005')

