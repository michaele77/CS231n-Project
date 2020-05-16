#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:37:24 2020

@author: ershov
"""


#Code taken from COCO demo example
#Modified to adapt from ipyb notebook

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import scipy



def imSeg(fnc_image, fnc_SegNum, fnc_inPlace = False):
    fnc_bordArr = np.linspace(0,255,fnc_SegNum + 1)
    print(fnc_bordArr)
    if fnc_inPlace:
        for fnc_iter in range(fnc_SegNum):
            boolArr = np.where((fnc_image > fnc_bordArr[fnc_iter]) & \
                               (fnc_image < fnc_bordArr[fnc_iter + 1]), True, False)
            fnc_image[boolArr] = sum(fnc_bordArr[fnc_iter], fnc_bordArr[fnc_iter + 1])//2
                      
        return
    else:
        fnc_tempArr = fnc_image.copy()
        for fnc_iter in range(fnc_SegNum):
            print(fnc_iter)
            boolArr = np.where((fnc_image >= fnc_bordArr[fnc_iter]) & \
                               (fnc_image <= fnc_bordArr[fnc_iter + 1]), True, False)
            print(boolArr)
            fnc_tempArr[boolArr] = (fnc_bordArr[fnc_iter] + fnc_bordArr[fnc_iter + 1]) //2
                  
        return fnc_tempArr  


def iterSKetch(inputImg, lowBound = 200, hiBound = 240, smoothFact = 0.25):
    #I found that means between 200 and 245 give the best results
    #180 and lower is usually too dark and 250 and above is really sparse
    
    #By default, choose mean that is closest to the midpoint for output
    #return a flag that says whether the final result was within the bounds
    #(ie True if it was OUTSIDE of bounds, makes it easy to search for problems)
    
    
    
    IMW, IMH, _ = inputImg.shape
    simpN = IMW*IMH

    sigrArr = np.linspace(0.1,0.25,3)
    shadeArr = np.linspace(0.1,0.25,3)
    
    meanTracker = 0
    goalMean = (lowBound + hiBound)/2
    goalRange = hiBound - goalMean
    boundFlag = True

    for i in sigrArr:
        for j in shadeArr:
            tempSktch,_ = cv2.pencilSketch(inputImg, sigma_s=60, sigma_r=i, shade_factor=j)
            tempSktch = scipy.ndimage.gaussian_filter(tempSktch, sigma=smoothFact)

            tempMean = np.sum(tempSktch) / simpN
            
            if abs(tempMean - goalMean) < abs(meanTracker - goalMean):
                meanTracker = tempMean
                func_best = tempSktch
                
            
            if abs(tempMean - goalMean) < goalRange:
                boundFlag = False
                
    print(meanTracker)
    return func_best, boundFlag
    
        
    
        
    




dataType='val2017'
annFile='coco/annotations/instances_{}.json'.format(dataType)

#Create coco helper class
coco=COCO(annFile)


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [324158])
#imgIds = coco.getImgIds(imgIds = [324159])
imgA = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0] 


imgIds=sorted(coco.getImgIds())
imgId = imgIds[np.random.randint(len(imgIds))]
img = coco.loadImgs(imgId)[0]


I = io.imread(img['coco_url'])
print(I.shape)



##SKETCHING EXPIREMENTATION BELOW

newImg = I[:,:,::-1]
cv2.imwrite("temp_images/temp1.jpg", newImg.astype(np.float32))
cv2.imwrite("temp_images/temp2.jpg", I.astype(np.float32))

h,w,_ = newImg.shape
totN = h*w

#Now sketch it out
sketch_gray1, _ = cv2.pencilSketch(newImg, sigma_s=60, sigma_r=0.05, shade_factor=0.05)
sketch_gray1 = scipy.ndimage.gaussian_filter(sketch_gray1, sigma=0.25)
print('sk1 = ' + str(np.sum(sketch_gray1)/totN))

sketch_gray2, _ = cv2.pencilSketch(newImg, sigma_s=60, sigma_r=0.1, shade_factor=0.1)
sketch_gray2 = scipy.ndimage.gaussian_filter(sketch_gray2, sigma=0.25)
print('sk2 = ' + str(np.sum(sketch_gray2)/totN))

sketch_gray3, _ = cv2.pencilSketch(newImg, sigma_s=60, sigma_r=0.2, shade_factor=0.2)
sketch_gray3 = scipy.ndimage.gaussian_filter(sketch_gray3, sigma=0.25)
print('sk3 = ' + str(np.sum(sketch_gray3)/totN))

sketch_gray4, _ = cv2.pencilSketch(newImg, sigma_s=60, sigma_r=0.3, shade_factor=0.3)
sketch_gray4 = scipy.ndimage.gaussian_filter(sketch_gray4, sigma=0.25)
print('sk4 = ' + str(np.sum(sketch_gray4)/totN))

cv2.imwrite("temp_images/sktch1.jpg", sketch_gray1.astype(np.uint8))
cv2.imwrite("temp_images/sktch2.jpg", sketch_gray2.astype(np.uint8))
cv2.imwrite("temp_images/sktch3.jpg", sketch_gray3.astype(np.uint8))
cv2.imwrite("temp_images/sktch4.jpg", sketch_gray4.astype(np.uint8))


theImg, ans = iterSKetch(newImg, lowBound = 200, hiBound = 240)
print(ans)
cv2.imwrite("temp_images/TESTTESTTEST.jpg", theImg.astype(np.float32))


#Convert to simple B&W
grayscaleDetails = cv2.cvtColor(newImg, cv2.COLOR_RGB2GRAY)
cv2.imwrite("temp_images/greyImg.jpg", grayscaleDetails.astype(np.uint8))

cv2.imwrite("temp_images/testImage.jpg", newImg.astype(np.uint8))







#See segmentation below... it doesnt do much :(
#
#up_thresh = imSeg(sketch_gray, 5, False)
#down_thresh = imSeg(sketch_gray, 20, False)
#
#cv2.imwrite("temp_images/temp4.jpg", up_thresh.astype(np.float32))
#cv2.imwrite("temp_images/temp5.jpg", down_thresh.astype(np.float32))

##SKETCHING EXPIREMENTATION ABOVE







#plt.axis('off')
#plt.imshow(I)
#plt.show()
