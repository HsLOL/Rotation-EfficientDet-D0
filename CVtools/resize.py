# coding=utf-8
# 对裁剪后的图片进行resize，同时在此代码中还将不是large-vehicle和small-vehicle的去掉了
# 此时的labelTxt的输入和输出txt均是八点坐标，不是OpenCv格式！

import os
import cv2
import numpy as np
from tqdm import tqdm

# file tree
#
# target_size 缩放到多大
# root_path  --| images --| *.png / *.jpg 在ext变量位置改
#            --| labelTxt --| *.txt
#
#
# output_path --| images --| *.png / *.jpg 扩展名随root_path/images中的扩展名
#             --| labelTxt --| *.txt


def write_txt(content, file_name, outpath):
    with open(os.path.join(outpath, 'labelTxt/', file_name + '.txt'), 'w') as f_out:
        for i in range(len(content)):
            single_content = content[i]
            single_content[:] = map(str, single_content[:])
            outline = single_content[0] + ' ' + single_content[1] + ' ' + single_content[2] + ' ' + single_content[3]\
                      + ' ' + single_content[4] + ' ' + single_content[5] + ' ' + single_content[6] + ' ' + single_content[7]\
                      + ' ' + single_content[8] + ' ' + single_content[9]
            if i < len(content) - 1:
                f_out.write(outline)
                f_out.write('\n')
            else:
                f_out.write(outline)


def make_path(path):
    image_path = os.path.join(path, 'images')
    txt_path = os.path.join(path, 'labelTxt')
    if not os.path.exists(image_path):
        os.makedirs(image_path, exist_ok=True)

    if not os.path.exists(txt_path):
        os.makedirs(txt_path, exist_ok=True)


def rescale(inpath, outpath, target_size, ext, keep_ratio):

    # 创建文件夹
    make_path(outpath)

    image_list = os.listdir(os.path.join(inpath, 'images/'))
    txt_list = os.listdir(os.path.join(inpath, 'labelTxt/'))
    valid_file = []
    for idx in tqdm(range(len(image_list)), ncols=88):
        # 获取图片
        single_image = image_list[idx]
        single_image_basename = os.path.splitext(single_image)[0]
        image = cv2.imread(os.path.join(inpath, 'images/', single_image))
        h, w, c = image.shape

        # 获取标注位置信息
        file_content = []
        with open(os.path.join(inpath, 'labelTxt/', single_image_basename + '.txt'), 'r') as f_in:
            lines = f_in.readlines()
            for line in lines:
                line = line.strip().split(' ')
                if line[8] == 'large-vehicle' or line[8] == 'small-vehicle':
                    file_content.append(line)

        if len(file_content) == 0:
            valid_file.append(single_image)
        else:
            if keep_ratio:
                scale = float(target_size) / float(h)
                im = cv2.resize(image, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                im_scale = np.array([scale, scale, scale, scale])
            cv2.imwrite(os.path.join(outpath, 'images', single_image_basename + ext), im)

            for index in range(len(file_content)):
                single_content = file_content[index]
                single_content[:8] = map(lambda x: float(x) * im_scale[0], single_content[:8])

            write_txt(file_content, single_image_basename, outpath)
    print(f'无效图片：{valid_file}\n无效图片个数{len(valid_file)}')


if __name__ == '__main__':
    rescale(
        inpath=r'/home/fzh/DOTA_devkit_YOLO-master/Big_Rotation_dataset/',
        outpath=r'/home/fzh/DOTA_devkit_YOLO-master/Big_Rotation_dataset/Rescale/',
        target_size=768,
        ext='.jpg',
        keep_ratio=True
    )
