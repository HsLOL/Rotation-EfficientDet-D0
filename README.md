## A project for Rotaion Detector by Pytorch  
This is my first project to finish the rotation detection by PyTorch.  
## Update Log  
[2021-04-14] Start this project.  
[2021-04-14] Update the DOTA_toolkit, which is a general toolkit to help finishsome preprocess.  
[2021-04-29] Finished this project roughly, I find the current detector has the unsastisfactory performance, and I will try my best to sovle these problems soon.  
[2021-04-29] Even the current rotation detector has a lower performace, I have already updated the relevant code.  
[2021-05-04] Update the basic script.  
## The next step  
- [X] Update the evaluation code for rotation detector(Finished but no update).  
- [X] Update the result of the whole DOTA picture.  
- [X] Find and solve the problems that I find.  
- [ ] Finish the multi scale images for training.  
- [ ] Finally finish this repo completely.  
## The performance about the current detector  
<img src="https://github.com/HsLOL/Rotation-EfficientDet-D0/blob/master/ShowResult/showresult.jpg" width="600" height="600"/>  

<img src="https://github.com/HsLOL/Rotation-EfficientDet-D0/blob/master/ShowResult/Merged.jpg" width="600" height="600"/>


## How to start this project  
* Prepare the environment(conda is recommended)  
`1.conda create -n <env_name> python=3.6`  
`2.conda activate <env_name>`  
`3.you should install torch=1.7.0, torchvision=0.8.1, cudatoolkit=11.0, and you can to search command on PyTorch.org`  
`4.pip install -r requirements.txt to install revelant`  
* Compile and build the skew iou and rotaion nms module
before you enjoy the code, you should also create the skew iou module and rotation nms module after complile some C++ and cython files.  
## References  
[link1] (https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)  
[link2] (https://zhuanlan.zhihu.com/p/358072483)  
My work is based on these very awesome open source contribution!
