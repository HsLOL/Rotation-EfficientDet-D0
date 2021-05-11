## 保存做毕设实验过程中用到的对数据集处理的脚本  
`ImgSplit_multi_process.py` 用于对原始图片进行裁剪，但目前代码存在生成的txt文件可能为空，需要将空的txt文件手动删除，再进行后面的操作，有的txt文件里还存在其他的类别，后续在完善.  
`resize.py`对裁剪后的图片进行缩放，提高输入到模型中图片的分辨率.  
`Img_Augment.py`支持对DOTA图片的水平、垂直旋转以及角度旋转，进行数据增强，同时可以对增强的结果标注可视化检测是否成功转换.  
`DOTA2opencv.py`用于将8点坐标转换为OPENCV格式，并支持对转换后的标注格式可视化查看是否转换成功.  
`txt2json.py`用于将OPENCV格式的txt按照训练集和测试集转换成json文件，其中instances_train.json里面是所有的train数据,instances_val.json是所有的val数据