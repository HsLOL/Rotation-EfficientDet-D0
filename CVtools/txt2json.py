# coding=utf-8
#
# 该代码用于将分割后的txt文件转换为json文件
# 一个json文件中包含所有的训练集和验证集中所有txt文件内容

# file tree
# root_path --|images --| *.jpg 可以根据文件夹中的扩展名在line83进行修改
#           --|labelTxt --| *.txt
#
# output_path --|json --| *.json  所有txt文件中的内容均在.json文件中



import os
import cv2
import json
from tqdm import tqdm


def GetFileFromThisRootDir(dir, ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root, dirs, files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles


def ReadOpencvFormat(txtpath):
    # 用于读取OPENCV格式文件内容
    # 同时去除掉文件中可能存在不是 'large-vehicle' 和 'small-vehicle'的实例

    filecontent = []
    result = []
    with open(txtpath, 'r') as f_in:
        lines = f_in.readlines()
    for line in lines:
        line = line.strip().split(' ')
        filecontent.append(line)

    for content in filecontent:
        if content[5] == 'large-vehicle' or content[5] == 'small-vehicle':
            object_dict = {}
            info = list(map(lambda x: float(x), content[:5]))
            object_dict['info'] = info
            object_dict['area'] = info[2] * info[3]
            object_dict['cat'] = content[5]
            object_dict['diff'] = content[6]
            result.append(object_dict)
    return result


def custombasename(fullname):
    # return os.path.basename(os.path.splitext(fullname)[0])
    return os.path.basename(fullname).split('.')[0]


def COCOTrain(srcpath, dstpath, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []

    for idx, name in enumerate(cls_names):
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 0
    inst_count = 0

    with open(dstpath, 'w') as f_out:
        filenames = GetFileFromThisRootDir(labelparent)
        for file in tqdm(filenames, ncols=88, desc='COCOTrain()'):
            basename = custombasename(file)

            imgpath = os.path.join(imageparent, basename + '.jpg')
            img = cv2.imread(imgpath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.jpg'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            # print(f"{type(data_dict['images'])}")
            data_dict['images'].append(single_image)

            objects = ReadOpencvFormat(os.path.join(labelparent, file))
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['cat']) + 1
                # single_obj['segmentation'] = []
                # single_obj['segmentation'].append(obj['info'])
                single_obj['segmentation'] = obj['info']
                single_obj['iscrowd'] = 0
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


def COCOVal(srcpath, dstpath, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []

    for idx, name in enumerate(cls_names):
        single_cat = {'id': idx + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 0
    inst_count = 0

    with open(dstpath, 'w') as f_out:
        filenames = GetFileFromThisRootDir(labelparent)
        for file in tqdm(filenames, ncols=88, desc='COCOVal()'):
            basename = custombasename(file)

            imgpath = os.path.join(imageparent, basename + '.jpg')
            img = cv2.imread(imgpath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.jpg'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            # print(f"{type(data_dict['images'])}")
            data_dict['images'].append(single_image)

            objects = ReadOpencvFormat(os.path.join(labelparent, file))
            for obj in objects:
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['cat']) + 1
                # single_obj['segmentation'] = []
                # single_obj['segmentation'].append(obj['info'])
                single_obj['segmentation'] = obj['info']
                single_obj['iscrowd'] = 0
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


if __name__ == '__main__':

    class_names = ['large-vehicle', 'small-vehicle']


    # COCOTrain(r'root_path',
    #           r'output_path'
    #           class_names)
    #
    #
    # COCOTrain(r'/home/fzh/DOTA_devkit_YOLO-master/Big_Rotation_dataset/还未转换成json格式/train',
    #           r'/home/fzh/DOTA_devkit_YOLO-master/Big_Rotation_dataset/还未转换成json格式/train/instances_train.json',
    #           class_names)
    #
    COCOVal(r'/home/fzh/DOTA_devkit_YOLO-master/Big_Rotation_dataset/还未转换成json格式/val',
            r'/home/fzh/DOTA_devkit_YOLO-master/Big_Rotation_dataset/还未转换成json格式/val/instances_val.json',
            class_names)
