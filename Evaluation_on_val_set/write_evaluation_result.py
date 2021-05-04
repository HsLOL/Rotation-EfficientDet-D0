# coding=utf-8
# 代码用于产生 result_classname/、row_DOTA_labels/、imgnamefile 文件夹及文件中的内容

import json
import os


def load_json(path):
    with open(path, 'r') as f_in:
        content = json.load(f_in)

    # 获取json文件中共有多少张图片
    fileimages = content['images']
    image_lists = []
    for idx in range(len(fileimages)):
        single_image = fileimages[idx]
        image_lists.append(os.path.splitext(single_image['file_name'])[0])

    return image_lists


def write_imgnamefile():
    ...


def write_file(path, img_lists):
    print(len(img_lists))
    with open(path, 'w') as f_out:
        for idx in range(len(img_lists)):
            if idx < len(img_lists) - 1:
                f_out.write(img_lists[idx])
                f_out.write('\n')
            if idx == len(img_lists) - 1:
                f_out.write(img_lists[idx])


def check_and_write_anno_file(path, txt_path, lists):
    head = ['imagesource:GoogleEarth', 'gsd:0.115726939386']
    invalid = []
    print(len(lists))
    for idx in range(len(lists)):
        flag = 1
        txtfile_name = os.path.join(path, lists[idx] + '.txt')
        with open(txtfile_name, 'r') as f_in:
            file_content = []
            # while True:
            #     line = f_in.readline()
            #     if line:
            #         split_lines = line.strip().split(' ')
            #         if len(split_lines) < 10:
            #             invalid.append(txtfile_name)
            #             continue
            #
            #         if split_lines[8] == 'small-vehicle' or split_lines[8] == 'large-vehicle':
            #             split_lines[:8] = map(lambda x: float(x), split_lines[:8])
            #             file_content.append(split_lines)
            #         else:
            #             continue
            #     # print(os.path.join(txt_path, lists[idx] + '.txt'))
            #     else:
            #         break
            lines = f_in.readlines()
            for i in range(len(lines)):
                line = lines[i]
                split_lines = line.strip().split(' ')
                if len(split_lines) < 10:
                    invalid.append(txtfile_name)
                    flag = 0
                    continue
                if split_lines[8] == 'small-vehicle' or split_lines[8] == 'large-vehicle':
                       split_lines[:8] = map(lambda x: float(x), split_lines[:8])
                       file_content.append(split_lines)
                else:
                    continue
            if flag == 1:
                with open(os.path.join(txt_path, lists[idx] + '.txt'), 'w') as f_out:
                    f_out.write(head[0] + '\n')
                    f_out.write(head[1] + '\n')
                    for i in range(len(file_content)):
                        strline = str(file_content[i][0]) + ' ' + str(file_content[i][1]) + ' ' + str(file_content[i][2]) + ' ' \
                        + str(file_content[i][3]) + ' ' + str(file_content[i][4]) + ' ' + str(file_content[i][5]) + ' ' \
                        + str(file_content[i][6]) + ' ' + str(file_content[i][7]) + ' ' + file_content[i][8] + ' ' + file_content[i][9]
                        f_out.write(strline)
                        if i < len(file_content) - 1:
                            f_out.write('\n')
    return invalid


if __name__ == '__main__':
    json_path = r'/home/fzh/Rotation-EfficinetDet/datasets/rotation_vehicles/annotations/instances_val.json'
    image_lists = load_json(json_path)

    # imgnamefile_path = r'/home/fzh/Rotation-EfficinetDet/Evaluation_on_val_set/imgnamefile.txt'
    # write_file(imgnamefile_path, image_lists)

    # 训练集和验证集所有图片对应的标注文件txt文件所在路径(num=1547)
    txt_path = r'/home/fzh/DOTA_devkit_YOLO-master/EfficientDet/labelTxt'
    dst_path = r'/home/fzh/Rotation-EfficinetDet/Evaluation_on_val_set/row_DOTA_labels'

    invalid_list = check_and_write_anno_file(txt_path, dst_path, image_lists)
    print(f"无效的文件：{invalid_list}")




