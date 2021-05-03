# coding=utf-8
'''
2019.10.8  ming71
功能:  对box进行anchor的kmeans聚类;
      输入的box格式支持voc， hrsc， yolo三种
注意:
    - 停止条件是最小值索引不变而不是最小值不变，会造成早停，可以改
    - 暂时仅支持voc标注,如需改动再重写get_all_boxes函数即可
评价方法：
    - anchor聚类采用iou评价 / 可视化(method1情况下)
    - area和ratio聚类采用可视化散点图
usage:
# 1. 如果可视化，聚类anchor时需使用method1
# 2. 如果想保存txt以及分析anchor情况，使用method2
# 3. 默认保存方式是yolo的格式，支持输出np.loadtxt的格式，在对应函数中打开注释即可
'''


# 目前该代码对应的文件结构：
# label_path --| *.txt
#            --| *.jpg
#
# save_path  --| 保存txt文件的路径


import numpy as np
import glob
import os
import cv2
from decimal import Decimal
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## a sample for kmeans via sklearn:
# import numpy as np
# from sklearn.cluster import KMeans
# data = np.random.rand(100, 3) #生成一个随机数据，样本大小为100, 特征数为3
# import ipdb; ipdb.set_trace()
# #假如我要构造一个聚类数为3的聚类器
# estimator = KMeans(n_clusters=3)#构造聚类器
# estimator.fit(data)#聚类
# label_pred = estimator.labels_ #获取聚类标签
# centroids = estimator.cluster_centers_ #获取聚类中心
# inertia = estimator.inertia_ # 获取聚类准则的总和

class Kmeans:
    def __init__(self, cluster_number, all_boxes, save_path=None, vis=None):
        self.cluster_number = cluster_number
        self.all_boxes = all_boxes
        self.save_path = save_path
        self.vis = vis

    # 输入两个二维数组:所有box和种子点box
    # 输出[num_boxes, k]的结果
    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number  # 类别

        box_area = boxes[:, 0] * boxes[:, 1]  # 列表切片操作：取所有行0列和1列相乘 ，得到gt的面积的行向量
        box_area = box_area.repeat(k)  # 行向量进行重复
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]  # 种子点的面积行向量
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area + 1e-16)
        assert (result > 0).all() == True, 'negtive anchors present , cluster again!'
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def result2txt(self, data):
        f = open(self.save_path, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            ### yolo format
            # if i == 0:
            #     x_y = "%d,%d" % (data[i][0], data[i][1])
            # else:
            #     x_y = ", %d,%d" % (data[i][0], data[i][1])
            x_y = "%d  %d \n" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()  # 最终输出的是w1,h1,w2,h2,w3,h3,...

    def anchor_clusters(self, size=None):

        boxes = np.array(self.all_boxes)  # 返回全部gt的宽高二维数组
        k = self.cluster_number
        ############   K-means聚类计算  ######
        #####  Method 1 : sklearn implemention
        # estimator = KMeans(n_clusters=k)
        # estimator.fit(boxes)             #聚类
        # label_pred = estimator.labels_   #获取聚类标签
        # centroids = estimator.cluster_centers_ #获取聚类中心
        # centroids = np.array(centroids)
        # result = centroids[np.lexsort(centroids.T[0, None])]              #将得到的三个anchor按照宽进行从小到大，重新排序
        # print("K anchors:\n {}\n".format(result))
        # print("Accuracy: {:.2f}%\n".format(self.avg_iou(boxes, result) * 100))
        # plt.figure('anchor_clusters')
        # plt.scatter(boxes[:,0], boxes[:,1], marker='.',c=label_pred)
        # plt.xlabel('anchor_w')
        # plt.ylabel('anchor_h')
        # plt.title('anchor_clusters')
        # for c in centroids:
        #     plt.annotate(s='cluster' ,xy=c ,xytext=c-20,arrowprops=dict(facecolor='red',width=3,headwidth = 6))
        #     plt.scatter(c[0], c[1], marker='*',c='red',s=100)

        #####  Method 2 : 自己写一边kmeans，这个的iou更高，推荐使用
        # #注意：这里代码选择的停止聚类的条件是最小值的索引不变，而不是种子点的数值不变。这样的误差理论会大一点，其实关系不大。
        box_number = boxes.shape[0]  # box个数
        distances = np.empty((box_number, k))  # 初始化[box_number , k]二维数组，存放自定义iou距离（obj*anchor）
        last_nearest = np.zeros((box_number,))  # [box_number , ]的标量
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # 种子点随机初始化

        # 种子点一旦重复会有计算错误,避免!
        while True:
            uniques_clusters = np.unique(clusters, axis=0)
            if len(uniques_clusters) == len(clusters):
                break
            clusters = boxes[np.random.choice(box_number, k, replace=False)]

        # k-means
        while True:
            # 每轮循环，计算种子点外所有点各自到k个种子点的自定义距离，并且按照距离各个点找离自己最近的种子点进行归类；计算新的各类中心；然后下一轮循环
            distances = 1 - self.iou(boxes, clusters)  # iou越大,距离越小

            current_nearest = np.argmin(distances, axis=1)  # 展开为box_number长度向量,代表每个box当前属于哪个种子点类别(0,k-1)
            if (last_nearest == current_nearest).all():  # 每个box的当前类别所属和上一次相同,不再移动聚类
                break

                # 计算新的k个种子点坐标
            for cluster in range(k):
                clusters[cluster] = np.median(boxes[current_nearest == cluster], axis=0)  # 只对还需要聚类的种子点进行位移
            last_nearest = current_nearest
        result = clusters[np.lexsort(clusters.T[0, None])]  # 将得到的三个anchor按照宽进行从小到大，重新排序
        print('\n-----anchor_cluster-----\n')
        print("K anchors:\n {}\n".format(result))
        print("Accuracy: {:.2f}%\n".format(self.avg_iou(boxes, result) * 100))

        if self.save_path:
            self.result2txt(result)

            ## 聚类结果分析(仅支持yolo格式的anchor文本)
            with open(self.save_path, 'r') as f:
                contents = f.read()
                w = list(map(int, contents.split(',')[::2]))
                h = list(map(int, contents.split(',')[1::2]))
                anchors = [anchor for anchor in zip(w, h)]
                ratio = [Decimal(anchor[0] / anchor[1]).quantize(Decimal('0.00')) for anchor in anchors]
                ratio.sort()
                area = [Decimal(anchor[0] * anchor[1]).quantize(Decimal('0.00')) for anchor in anchors]
                area.sort()
                #####   自定义需要分析的数据  ###
                squre = [float(s) ** 0.5 for s in area]
                print('ratio:\n{}\n\narea:\n{}\n'.format(ratio, area))
                print('sqrt(area):\n{}'.format(squre))

    # 面积聚类
    def area_cluster(self):
        boxes = np.array(self.all_boxes)
        areas = boxes[:, 0] * boxes[:, 1]

        estimator = KMeans(n_clusters=self.cluster_number)  # 新建Kmeans对象，并传入参数
        estimator.fit(areas.reshape(-1, 1))  # 聚类，即开始迭代
        label_pred = estimator.labels_  # 获取聚类标签
        # print(f"聚类标签:{label_pred}\n聚类标签的个数:{len(label_pred)}")  # 对所有宽高比属于哪个簇分配标签
        centroids = estimator.cluster_centers_  # 获取聚类中心
        # print(f"聚类中心:{centroids}\n聚类中心的个数:{len(centroids)}")  # 获得聚类的中心对应的宽高比
        centroids = centroids[np.lexsort(centroids.T)]  # 排个序
        print(f"对centroids排序结果:{centroids}")
        centroids = np.array([int(i) for i in centroids]).reshape(-1, 1)  # 取个整
        print(f"对centroids取整后的结果:{centroids}")
        print('\n-----area_cluster-----\n')
        print(centroids)
        if self.vis:
            plt.figure('area_cluster')
            plt.scatter(range(len(areas)), areas.squeeze(), marker='.', c=label_pred)
            plt.xlabel('gt_num')
            plt.ylabel('area')
            plt.title('area_cluster')
            for c in centroids:
                xy = np.array([int(0.5 * len(boxes)), c.item()])
                plt.scatter(int(0.5 * len(boxes)), c.item(), marker='*', c='red', s=100)

    # 宽高比聚类
    def ratio_cluster(self, vis=False):
        boxes = np.array(self.all_boxes)
        ratios = boxes[:, 0] / boxes[:, 1]  # ration = w / h

        estimator = KMeans(n_clusters=self.cluster_number)
        estimator.fit(ratios.reshape(-1, 1))  # 聚类
        label_pred = estimator.labels_  # 获取聚类标签
        centroids = estimator.cluster_centers_  # 获取聚类中心
        centroids = centroids[np.lexsort(centroids.T)]  # 排个序(从小到大)
        # 表示为分子或分母1便于直观观察
        print('\n-----ratio_cluster-----\n')
        for i, c in enumerate(centroids):
            num, den = c.item().as_integer_ratio()  # 根据给定的小数，利用as_integer_ratio()的方法，得到是哪两个整数的比值
            if c > 1: num /= den; den = 1; num = Decimal(num).quantize(Decimal('0.00'))
            if c < 1: den /= num; num = 1; den = Decimal(den).quantize(Decimal('0.00'))
            ratio = str(num) + '/' + str(den)
            print(ratio)
        if self.vis:
            plt.figure('ratio_cluster')
            plt.scatter(range(len(ratios)), ratios.squeeze(), marker='.', c=label_pred)
            plt.xlabel('gt_num')
            plt.ylabel('ratio')
            plt.title('ratio_cluster')
            for c in centroids:
                xy = np.array([int(0.5 * len(boxes)), c.item()])
                plt.scatter(int(0.5 * len(boxes)), c.item(), marker='*', c='red', s=100)

    # gt和图像的面积占比聚类 [只有yolo格式支持]
    def img_proportion_cluster(self, vis=True):
        boxes = np.array(self.all_boxes)
        self.area_cluster()


# 返回所有label的box,形式为[[w1,h1],[w2,h2],...]
# 目前只有yolo支持resize聚类，其他的很容易实现，后面要用再写
def get_all_boxes(path, mode=None, resize=False):
    assert not mode is None, 'Input correct label mode,such as : voc, hrsc, yolo'
    boxes = []

    if mode == 'voc':
        labels = sorted(glob.glob(os.path.join(path, '*.*')))
        for label in labels:
            with open(label, 'r') as f:
                contents = f.read()
                objects = contents.split('<object>')
                if len(objects) == 0: pass

                for object in objects:
                    xmin = int(float(object[object.find('<xmin>') + 6: object.find('</xmin>')]))
                    xmax = int(float(object[object.find('<xmax>') + 6: object.find('</xmax>')]))
                    ymin = int(float(object[object.find('<ymin>') + 6: object.find('</ymin>')]))
                    ymax = int(float(object[object.find('<ymax>') + 6: object.find('</ymax>')]))

                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    boxes.append((box_w, box_h))

    elif mode == 'hrsc':  # xml格式
        rotate = True
        labels = sorted(glob.glob(os.path.join(path, '*.*')))
        for label in labels:
            with open(label, 'r') as f:
                contents = f.read()
                objects = contents.split('<HRSC_Object>')
                objects.pop(0)
                if len(objects) == 0: pass

                for object in objects:
                    if not rotate:
                        xmin = int(object[object.find('<box_xmin>') + 10: object.find('</box_xmin>')])
                        ymin = int(object[object.find('<box_ymin>') + 10: object.find('</box_ymin>')])
                        xmax = int(object[object.find('<box_xmax>') + 10: object.find('</box_xmax>')])
                        ymax = int(object[object.find('<box_ymax>') + 10: object.find('</box_ymax>')])
                        box_w = xmax - xmin
                        box_h = ymax - ymin
                    else:  # 旋转框
                        box_w = int(float(object[object.find('<mbox_w>') + 8: object.find('</mbox_w>')]))
                        box_h = int(float(object[object.find('<mbox_h>') + 8: object.find('</mbox_h>')]))
                    boxes.append((box_w, box_h))

    elif mode == 'yolo':
        # label_path 里面就是.txt文件
        return_ratio = True  # 只返回比例,供数据分析用
        labels = sorted(glob.glob(os.path.join(path, '*.txt*')))
        for label in tqdm(labels, desc='Loading labels'):
            img_path = os.path.join(os.path.split(label)[0], os.path.split(label)[1][:-4] + '.jpg')
            height, width, _ = cv2.imread(img_path).shape
            with open(label, 'r') as f:
                contents = f.read()
                lines = contents.split('\n')
                lines = [x for x in contents.split('\n') if x]  # 移除空格
                for object in lines:
                    coors = object.split(' ')
                    if not return_ratio:
                        box_w = int(float(coors[3]) * width)
                        box_h = int(float(coors[4]) * height)

                        if not resize:
                            boxes.append((box_w, box_h))
                        else:  # 长边resize到固定尺度
                            stride = max(height, width) / resize
                            boxes.append((box_w / stride, box_h / stride))
                    else:
                        box_w = float(coors[3])
                        box_h = float(coors[4])
                        boxes.append((box_w, box_h))

    else:
        print('Unrecognized label mode!!')
    return boxes


if __name__ == "__main__":
    cluster_number = 3  # 种子点个数
    vis = True  # 可视化聚类结果
    resize = 416  # 长边缩放到416聚类
    label_path = '/home/fzh/Kmeans/COCO/train/labels/train2017'
    save_path = '/home/fzh/Kmeans/COCO/12.txt'
    # all_boxes:存储所有.txt文件中实例的长度和宽度(先宽度后长度)
    all_boxes = get_all_boxes(label_path, 'yolo', resize=resize)
    kmeans = Kmeans(cluster_number, all_boxes,
                    save_path=save_path,
                    vis=vis)

    kmeans.anchor_clusters()
    kmeans.area_cluster()
    kmeans.ratio_cluster()
    kmeans.img_proportion_cluster()

    if vis:
        plt.show()

'''
K anchors:
 [[153.87419355  31.30967742]
 [343.26412214  52.12366412]
 [574.57024793 109.7231405 ]]
Accuracy: 69.19%
-----area_cluster-----
[[ 12130]
 [ 42951]
 [113378]]
-----ratio_cluster-----
4.18/1
6.50/1
8.75/1
'''