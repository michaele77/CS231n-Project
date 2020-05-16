#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:49:06 2020

@author: ershov
"""


from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import scipy
import os


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
                
    return func_best, boundFlag


#Helper function for retrieving the image to make the code cleaner
def getCocoImg(givenID):
    func_img = coco.loadImgs(givenID)[0]
    temp_funcImg = io.imread(func_img['coco_url'])
    out = temp_funcImg
    if len(temp_funcImg.shape) == 3:
        out = temp_funcImg[:,:,::-1]
    return out #reverse the fucking BGR





##Start of main()
print('Starting data creation...')
print('~~~~~~~~')
print('~~~~~~~~')

#Define image directories where they will be deposited
file_og = 'training_images/Original/'
file_sk = 'training_images/Sketch/'
file_bw = 'training_images/Grayscale/'

fext = '.jpg'


dataType='val2017'
annFile='coco/annotations/instances_{}.json'.format(dataType)

#Create coco helper class
coco=COCO(annFile)

#Get all relevant image ids from the dataset, get directory lists
imgIds = sorted(coco.getImgIds())
#imgIds = imgIds[0:20]

imageNum = len(imgIds)
ogDirList = os.listdir(file_og)
skDirList = os.listdir(file_sk)
bwDirList = os.listdir(file_bw)

#remove file extensions and convert to an int for each string
#assumes .jpg format or other .XXX format
#assumes data is given as xx_[number].jpg, where [number] is extracted
ogDirList.remove('.DS_Store')
skDirList.remove('.DS_Store')
bwDirList.remove('.DS_Store')

if ogDirList:
    ogDirList = [int(i[2:-4]) for i in ogDirList]
    skDirList = [int(i[2:-4]) for i in skDirList]
    bwDirList = [int(i[2:-4]) for i in bwDirList]


#Counters for the loop
outOfBounds = 0
numImgs = 0
printEvery = 10
preserveOlds = True
totImages = 0

print('Pre-Initialization finished, starting main loop')
print(str(imageNum) + ' images to finish')
print('~~~~~~~~')
print('~~~~~~~~')

for i in imgIds:
    totImages += 1
    
    #Check if the directory is empty before
    if i in ogDirList and preserveOlds:
        continue
        
        
    currImg = getCocoImg(i)
    
    #If BW, skip them
    if len(currImg.shape) == 2:
        continue;
    
    gsTemp = cv2.cvtColor(currImg, cv2.COLOR_RGB2GRAY)
    skTemp, outBnds = iterSKetch(currImg, lowBound = 220, hiBound = 245)
    
    cv2.imwrite(file_og + 'og' + str(i) + fext, currImg.astype(np.float32))
    cv2.imwrite(file_bw + 'bw' + str(i) + fext, gsTemp.astype(np.float32))
    cv2.imwrite(file_sk + 'sk' + str(i) + fext, skTemp.astype(np.float32))
    
    if outBnds:
        outOfBounds += 1
        
    numImgs += 1
    if numImgs % printEvery == 0:
        print('Generated ' + str(numImgs) + ' images, ran through ' + \
              str(totImages) + ' images, but ' + str(outOfBounds) + \
              ' were out of bounds')
    
        
    




