## A project for Rotaion Detector by Pytorch  
## Todo List  
* Reimplentation /Efficientdet/dataset.py(as well as training_params and val_params)  
* Make the Rotation Dataset and make the anno file to json file  
* Modified the efficientdet/model.Regressor() line 374 self.header() num_anchors x 5(x_c, y_c, width, height, theta)  
* Follow the third step, should also notice the Regressor().forward() line 386-393  
* Delete the `label smoothing` trick  
* Modify the class Anchor() to add the parameter theta in utls.py(line 55)
* Add poly_iou loss.py line 102  
* calculate the regression loss in the loss.py line 181  
* Maybe also modified the smooth L1 loss to calculate the regression loss(loss.py line 183-187)  
* 
