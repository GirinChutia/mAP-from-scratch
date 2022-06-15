import cv2
import numpy as np
import matplotlib.pyplot as plt

class FormatResult:
    def __init__(self,torch_result):
        self.np_xyxy = torch_result.cpu().numpy()
        self._total = len(self.np_xyxy)
        
    def get_coordinates(self,array):
        x0,y0 = array[0],array[1]
        x1,y1 = array[2],array[3]
        return [int(x0),int(y0),int(x1),int(y1)]

    def get_confidence(self,array):
        conf = array[4]
        return conf
    
    def get_class(self,array):
        cl = array[5]
        return int(cl)

    def return_result(self,array):
        coord = self.get_coordinates(array)
        conf = self.get_confidence(array)
        cl = self.get_class(array)
        return coord,conf,cl

    def return_all_results(self):
        cords = []
        conf = []
        clss = []
        for item in range(self._total):
            cords.append(self.return_result(self.np_xyxy[item])[0])
            conf.append(self.return_result(self.np_xyxy[item])[1])
            clss.append(self.return_result(self.np_xyxy[item])[2])
        return cords,conf,clss

def filter_classes(coordinates,confidence,classes,filter_labels=[]):
    new_cd,new_cf,new_cl = [],[],[]
    for cd,cf,cl in zip(coordinates,confidence,classes):
        if cl in filter_labels:
            new_cd.append(cd)
            new_cf.append(cf)
            new_cl.append(cl)
    return new_cd,new_cf,new_cl

def boxes_from_yoloformat(imagePath,
                          labelPath,
                          output_format='xyxy'): 
    image = cv2.imread(imagePath) 
    (hI, wI) = image.shape[:2] 
    lines = [line.rstrip('\n') for line in open(labelPath)] 
    boxes = [] 
    if lines != ['']: 
        for line in lines: 
            components = line.split(" ") 
            category = components[0] 
            x  = int(float(components[1])*wI - float(components[3])*wI/2) 
            y = int(float(components[2])*hI - float(components[4])*hI/2) 
            h = int(float(components[4])*hI) 
            w = int(float(components[3])*wI) 
            x_min,y_min,x_max,y_max = x,y,x+w,y+h
            if output_format == 'xywh':
                boxes.append((category, (x, y, w, h))) 
            if output_format == 'xyxy':
                boxes.append((category, (x_min,y_min,x_max,y_max)))     
    return (image,boxes)

def showBoxes(image,gt_boxes,pd_boxes): 
    plt.figure(figsize=(12,12))
    gtImg = image.copy() 
    pdImg = image.copy()
    categoriesColors = {0: (255,100,0),1:(0,0,255)} 
    for box in gt_boxes: 
        (category, (x1, y1, x2, y2)) = box
        if int(category) in categoriesColors.keys(): 
            cv2.rectangle(gtImg,(x1,y1),(x2,y2),categoriesColors[int(category)],5) 
        else: 
            cv2.rectangle(gtImg,(x1,y1),(x2,y2),(0,255,0),5) 
    for box in pd_boxes: 
        (category, (x1, y1, x2, y2)) = box
        if int(category) in categoriesColors.keys(): 
            cv2.rectangle(pdImg,(x1,y1),(x2,y2),categoriesColors[int(category)],5) 
        else: 
            cv2.rectangle(pdImg,(x1,y1),(x2,y2),(0,255,0),5) 
    
    plt.subplot(1,2,1)
    plt.imshow(gtImg[:,:,::-1]) 
    plt.title("Ground Truth")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(pdImg[:,:,::-1]) 
    plt.title("Predictions")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def get_iou(bb1,bb2):
    '''
    bb format [x_top_left,y_top_left,x_botton_right,y_botton_right]
    '''
    xleft = max(bb1[0], bb2[0])
    ytop = max(bb1[1], bb2[1])
    xright = min(bb1[2], bb2[2])
    ybottom = min(bb1[3], bb2[3])
    
    if (xright < xleft) or (ybottom < ytop):
        return 0
    
    area_of_intersection = (xright - xleft + 1)*(ybottom - ytop + 1)
    
    bb1_area = (bb1[2]-bb1[0] + 1) * (bb1[3]-bb1[1] + 1)
    bb2_area = (bb2[2]-bb2[0] + 1) * (bb2[3]-bb2[1] + 1)
    
    _iou = (area_of_intersection/(bb1_area + bb2_area-area_of_intersection))
    
    return _iou

        
    