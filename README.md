## A project for Rotaion Detector by Pytorch  
This is my first project to finish the rotation detection by PyTorch.  
## Update Log  
[2021-04-14] Start this project.  
[2021-04-14] Update the DOTA_toolkit, which is a general toolkit to help finish some preprocess.  
[2021-04-29] Finished this project roughly, I find the current detector has the unsastisfactory performance, and I will use the K-means to solve the problem.  
[2021-04-29] Even the current rotation detector with the K-meas has the map=0.62 for the `small-vehicle` and `large-vehicle`, I have already updated the relevant code.  
[2021-05-04] Update the basic script.   
## The performance about the current detector  
<img src="https://github.com/HsLOL/Rotation-EfficientDet-D0/blob/master/ShowResult/showresult.jpg" width="450" height="450"/>  

<img src="https://github.com/HsLOL/Rotation-EfficientDet-D0/blob/master/ShowResult/Merged.jpg" width="450" height="450"/>


## How to start this project  
#### 1. Prepare the environment(conda is recommended)  
```
conda create -n <env_name> python=3.6  
conda activate <env_name>  
you should install torch=1.7.0, torchvision=0.8.1, cudatoolkit=11.0, and you can to search command on PyTorch.org
pip install -r requirements.txt to install revelant  
```  
**Note**: If you meet some troubles about the environment, you can check the check.txt  
#### 2. Compile and build the skew iou and rotaion nms module
Before you run this project, you should create the skew iou module and rotation nms module after compliling some C++ and cython files.  
#### 2.1. Install polyiou module  
the polyiou is used to calculate the skew iou and you can follow these steps to install it.  
```
cd polyiou  
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```  
#### 2.2. Install rotation nms module  
the rotation nms is used in the inference step, and you can follow these steps to install it.  
```
cd utils/nms
make
```  
#### 3. Make the Dataset
```you can reference the dataset format through the link1```
#### 4. Prepare the weight file  
```you can get the weight file(EfficientDet-D0) from the link1```
## References  
[link1] (https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)  
[link2] (https://zhuanlan.zhihu.com/p/358072483)  
My work is based on these very awesome open source contribution!
