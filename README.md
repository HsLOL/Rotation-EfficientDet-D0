## A project for Rotaion Detector by Pytorch  
## Todo List  
- [X] Reimplentation /Efficientdet/dataset.py(as well as training_params and val_params)(maybe have finished:))  
- [X] Make the Rotation Dataset and make the anno file to json file(Maybe has finished:))  
- [X] Modified the efficientdet/model.Regressor() line 374 self.header() num_anchors x 5(x_c, y_c, width, height, theta)  
- [X] Follow the third step, should also notice the Regressor().forward() line 386-393  
- [X] Delete the `label smoothing` trick  
- [X] Modify the class Anchor() to add the parameter theta in utls.py(line 55)
- [X] Add poly_iou loss.py line 102  
- [X] calculate the regression loss in the loss.py line 181  
- [X] Maybe also modified the smooth L1 loss to calculate the regression loss(loss.py line 183-187)-->modified the regression loss to smooth L1 loss.
## The Next Step  
* Finish the understanding and coding of skew IOU  
* Finish the complete project about Rotation Detector
