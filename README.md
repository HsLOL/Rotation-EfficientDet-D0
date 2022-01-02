## :rocket:A Rotation Detector based EfficientDet PyTorch  
This is a rotation detector pytorch implementation based on EfficientDet horizontal detector.  
The pytorch re-implement efficientdet is [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch).  
Original paper link is [here](https://arxiv.org/abs/1911.09070).
## :pushpin:Update Log  
[2021-04-14] Start this project.  
[2021-04-29] Finish this project roughly and add K-means algorithm.    
[2021-05-04] Upload revelant scripts.  
[2022-01-02] Perfect this repo and hope it will be helpful for some users who want to learn rotation detection.  
## Performance of the implemented Rotation Detector  
### Detection Performance on Small image.
<img src="https://github.com/HsLOL/Rotation-EfficientDet-D0/blob/master/pics/showresult.jpg" width="300" height="300"/>  

### Detection Performance on Big image. 
<img src="https://github.com/HsLOL/Rotation-EfficientDet-D0/blob/master/pics/Merged.jpg" width="300" height="300"/>  

## Get Started
### Installation  
Install requirements:
```
conda create -n Rtdet python=3.7  
conda activate Rtdet  
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt  

Note: If you meet some troubles about installing environment, you can see the check.txt for more details.  
```
Install skew iou module:
```
cd polyiou
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
Install rotation nms module:
```
cd utils/nms
make
```
## Demo
you should download the trained weight file below and put the pth file into `log` folder.
```
# run the simple inference script to get detection result.
python show.py --img_path ./test/demo1.jpg --pth ./logs/rotation_vehicles/efficientdet-d0_48_3200.pth
```
## Train
### 1. Prepare dataset
```
# dataset structure should be like this
datasets/
    -your_project_name/
        -train_set_name/
            -*.jpg
        -val_set_name/
            -*.jpg
        -annotations
            -instances_{train_set_name}.json
            -instances_{val_set_name}.json

# for example, coco2017
datasets/
    -coco2017/
        -train2017/
            -000000000001.jpg
            -000000000002.jpg
            -000000000003.jpg
        -val2017/
            -000000000004.jpg
            -000000000005.jpg
            -000000000006.jpg
        -annotations
            -instances_train2017.json
            -instances_val2017.json
```
### 2. Manual set project's hyper parameters
```
# create a yml file {your_project_name}.yml under 'projects'folder
# modify it following 'coco.yml'

# for example
project_name: coco
train_set: train2017
val_set: val2017
num_gpus: 4  # 0 means using cpu, 1-N means using gpus

# mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

# this is coco anchors, change it if necessary
anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

# objects from all labels from your dataset with the order from your annotations.
# its index must match your dataset's category_id.
# category_id is one_indexed,
# for example, index of 'car' here is 2, while category_id of is 3
obj_list: ['person', 'bicycle', 'car', ...]

```
### 3.a. Train rotation detector on a custom dataset from scratch
```
# train rotation efficientdet-d0 on a custom dataset
# with batchsize 32 and learning rate 5e-3

python train.py -c 0 -p your_project_name --batch_size 32 --lr 5e-3
```
### 3.b. Train rotation detector with pretrained weight file which is trained on horizontal datasets.
```
# train rotation efficientdet-d0 on a custom dataset with pretrained weights which is trained on horizontal datasets.
# with batchsize 32 and learning rate 5e-3 for 10 epoches

python train.py -c 0 -p your_project_name --batch_size 32 --lr 5e-3 --num_epochs 10 \
 --load_weights /path/to/your/weights/efficientdet-d0.pth

# with a pretrained weight file, you can even freeze the backbone and train heads only
# to speed up training and help convergence.

python train.py -c 0 -p your_project_name --batch_size 32 --lr 5e-3 --num_epochs 10 \
 --load_weights /path/to/your/weights/efficientdet-d0.pth \
 --head_only True
```
### 4. Early stopping a training session.
```
# while training, press Ctrl+c, the program will catch KeyboardInterrupt
# and stop training, save current checkpoint.
```
### 5. Resume Training
```
# let says you started a training session like this.

python train.py -c 0 -p your_project_name --batch_size 32 --lr 5e-3 \
 --load_weights /path/to/your/weights/efficientdet-d0.pth \
 --head_only True

# then you stopped it with a Ctrl+c, it exited with a checkpoint

# now you want to resume training from the last checkpoint
# simply set load_weights to 'last'

python train.py -c 0 -p your_project_name --batch_size 16 --lr 1e-3 \
 --load_weights last \
 --head_only False
```
### Evaluation
```
```
