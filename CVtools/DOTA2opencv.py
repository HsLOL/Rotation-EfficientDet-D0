# coding=utf-8

# This code is used to transform DOTA format to opencv format
# And it's also used to visualize the GT anno in the original image

# file tree
# root_path --|
#             |--images --|
#                       --| *.png  扩展名可在line 136修改
#
#             |--labelTxt --|
#                         --|*.txt
#
# output_root_path --|
#                    |--labelTxt --|
#                                --|*.txt
#                    |--visualize_anno --|
#                                      --|*.png  扩展名可在line 169修改
#
import argparse
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm


def print_list(print_list):
    # 按行输出list中的内容
    for i in range(len(print_list)):
        print(print_list[i])


def check_theta_and_modify(rotation_list):
    # 检查OPENCV格式中的角度是否有问题,并进行记录和更改
    temp_rotation_list = rotation_list
    zero_index = []
    positive_index = []
    vertical_index = []
    for i in range(len(temp_rotation_list)):
        theta = float(temp_rotation_list[i][4])
        width, height = float(temp_rotation_list[i][2]), float(temp_rotation_list[i][3])

        if theta == 0:
            zero_index.append(i)
            temp_rotation_list[i][4] = float(temp_rotation_list[i][4]) - 90
            temp = width
            width = height
            height = temp

            temp_rotation_list[i][2] = width
            temp_rotation_list[i][3] = height

        if theta == -90.0:
            vertical_index.append(i)

        if theta > 0:
            positive_index.append(i)
    print(f'经过OPENCV自身函数转换后/检查后 theta=0的个数为{len(zero_index)}\ntheta>0的个数为{len(positive_index)}\ntheta=-90个数为{len(vertical_index)}')

    return temp_rotation_list, zero_index, positive_index, vertical_index


def is_positive(poly_list):
        flag = np.any(list(map(lambda x: x < 0, poly_list[:8])))
        return flag


def DOTA2OPCV(poly):
    results = []
    for idx in range(len(poly)):
        # 将instance超出边界的物体去除掉
        if not is_positive(poly[idx]):
            # ------------------------------------
            points1 = (poly[idx][0], poly[idx][1])
            points2 = (poly[idx][2], poly[idx][3])
            points3 = (poly[idx][4], poly[idx][5])
            points4 = (poly[idx][6], poly[idx][7])
            cls = poly[idx][8]
            diff = poly[idx][9]
            vertex = np.array([points1, points2, points3, points4], dtype=np.float32)
            x_ctr, y_ctr = cv2.minAreaRect(vertex)[0][0], cv2.minAreaRect(vertex)[0][1]
            width, height = cv2.minAreaRect(vertex)[1][0], cv2.minAreaRect(vertex)[1][1]
            theta = cv2.minAreaRect(vertex)[-1]
            results.append([x_ctr, y_ctr, width, height, theta, cls, diff])
    return results


def read_anno_and_output_file(args):
    labelTxt_path = os.path.join(args.root_path, 'labelTxt')
    txt_filelist = os.listdir(labelTxt_path)
    filename_list = [os.path.splitext(os.path.basename(x))[0] for x in txt_filelist]
    for idx, txtfile in tqdm((enumerate(txt_filelist)), ncols=88):
        with open(os.path.join(labelTxt_path, txtfile), 'r') as f_in:
            lines = f_in.readlines()
        # txt_head = lines[:2]
        txt_head = []
        txt_body = lines
        temp = []
        for index, line in enumerate(txt_body):
            each_line = line.strip().split(' ')  # ['76.0', '847.0', '76.0', '864.0', '34.0', '866.0', '34.0', '847.0', 'small-vehicle', '0']
            temp.append(list(map(float, each_line[:8])))
            temp[index].append(each_line[8])
            temp[index].append(each_line[9])
        Rotation_ann = DOTA2OPCV(temp)  # 格式中可能存在theta=0的问题
        modified_Rotation_ann, _, _, _ = check_theta_and_modify(Rotation_ann)
        # print_list(modified_Rotation_ann)

        with open(os.path.join(args.output_path, 'labelTxt', f'{txtfile}'), 'w') as f_out:
            for jdx in range(len(txt_head)):
                f_out.write(txt_head[jdx])
            for j in range(len(modified_Rotation_ann)):
                line_content = modified_Rotation_ann[j]  # [55.0, 856.5, 19.0, 42.0, -90.0, 'small-vehicle', '0']
                out_line = str(line_content[0]) + ' ' + str(line_content[1]) + ' ' + str(line_content[2]) \
                 + ' ' + str(line_content[3]) + ' ' + str(line_content[4]) + ' ' + line_content[5] + ' ' + line_content[6]
                f_out.write(out_line + '\n')


def OPENCV2xywh(opencv_list):
    poly_list = []
    for idx in range(len(opencv_list)):
        opencv_list[idx][:5] = map(float, opencv_list[idx][:5])
        x_c, y_c = int(opencv_list[idx][0]), int(opencv_list[idx][1])
        width, height = int(opencv_list[idx][2]), int(opencv_list[idx][3])
        theta = int(opencv_list[idx][4])
        rect = ((x_c, y_c), (width, height), theta)
        poly = np.float32(cv2.boxPoints(rect))
        poly_list.append(poly)
    return poly_list


def drawOPENCVformat(args):
    # 将OPENCV格式转换后的anno读取出来，进行可视化检查
    for visualize_name in tqdm(os.listdir(os.path.join(args.root_path, 'images')), ncols=88):
        basename_visualize_name = visualize_name.split('.')[0]
        img = cv2.imread(os.path.join(args.root_path, 'images', basename_visualize_name + '.jpg'))
        with open(os.path.join(args.output_path, 'labelTxt', basename_visualize_name + '.txt'), 'r') as f_in:
            lines = f_in.readlines()
            # head = lines[:2]  # 为了与原始代码代码保持一致
            head = []
            body = lines
            content = []
            for line in body:
                line = line.strip().split(' ')
                content.append(line)

            # check theta
            print(f'---- 检查经过编写函数转换后{basename_visualize_name}的OPENCV格式是否含有角度问题 ----')
            modified_list, zero_idx, positive_idx, vertical_idx = check_theta_and_modify(content)
            # print_list(modified_list)

            if len(zero_idx) + len(positive_idx) == 0:
                print('已经不存在角度问题')
            else:
                temp_zero_list = []
                for idx in zero_idx:
                    temp_zero_list.append(content[idx])
                print(f'获取的theta=0的标注格式: {print_list(temp_zero_list)}')

            rect = OPENCV2xywh(content)
            rect = np.int0(rect)
            rect = np.array(rect)

            cv2.drawContours(image=img,
                             contours=rect,
                             contourIdx=-1,
                             color=[0, 0, 255],
                             thickness=2)
            cv2.imwrite(os.path.join(args.output_path, 'visualize_anno', f'{basename_visualize_name}.jpg'), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default=r'/home/fzh/DOTA_devkit_YOLO-master/Big_Rotation_dataset/Data_Augment/Augment_vertical')

    parser.add_argument('--output_path', type=str,
                        default=r'/home/fzh/DOTA_devkit_YOLO-master/Big_Rotation_dataset/OpencvFormat')

    args = parser.parse_args()

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(os.path.join(args.output_path, 'labelTxt'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'visualize_anno'), exist_ok=True)

    # 下面两行只适用于先把八点坐标的txt转换成OPENCV后，进行可视化检查，
    read_anno_and_output_file(args)
    drawOPENCVformat(args)



