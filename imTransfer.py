#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:40:14 2020

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
import random


file_og = 'training_images/Original/'
file_sk = 'training_images/Sketch/'
file_bw = 'training_images/Grayscale/'

sourceA = file_sk
sourceB = file_og
sinkA = 'pix2pix/path/to/data/A/'
sinkB = 'pix2pix/path/to/data/B/'

testStr = 'test/'
valStr = 'val/'
trainStr = 'train/'

#Specify how many you want in each split 
splitTrain = 200
splitTest = 50
splitVal = 50





srcDirListA = os.listdir(sourceA)
srcDirListA.remove('.DS_Store')
prefA = srcDirListA[0][0:2]
srcDirListA = [int(i[2:-4]) for i in srcDirListA] #doesnt sort unless in int
srcDirListA.sort()
srcDirListA = [prefA + str(i) + '.jpg' for i in srcDirListA]

srcDirListB = os.listdir(sourceB)
srcDirListB.remove('.DS_Store')
prefB = srcDirListB[0][0:2]
srcDirListB = [int(i[2:-4]) for i in srcDirListB]
srcDirListB.sort()
srcDirListB = [prefB + str(i) + '.jpg' for i in srcDirListB]


#Now move and copy for each split
for i in range(0,splitTrain):
    try:
        #Assume B is colored, take out transpose if it isnt
        temp = cv2.imread(sourceB + srcDirListB[i])
        a = cv2.imwrite(sinkB + trainStr +\
                    srcDirListB[i][2:], temp.astype(np.float32))
        if not a:
            print('Couldnt write image in A')
        
    except:
        print('Found a greyscale in OG')
        continue
    
    
    #Assume B is greyscale, take out flag if it isnt
    temp = cv2.imread(sourceA + srcDirListA[i], 0)
    cv2.imwrite(sinkA + trainStr +\
                srcDirListA[i][2:], temp.astype(np.float32))

for i in range(splitTrain,splitVal + splitTrain):
    try:
        #Assume B is colored, take out transpose if it isnt
        temp = cv2.imread(sourceB + srcDirListB[i])
        a = cv2.imwrite(sinkB + valStr +\
                    srcDirListB[i][2:], temp.astype(np.float32))
        if not a:
            print('Couldnt write image in A')
        
    except:
        print('Found a greyscale in OG')
        continue
    
    
    #Assume B is greyscale, take out flag if it isnt
    temp = cv2.imread(sourceA + srcDirListA[i], 0)
    cv2.imwrite(sinkA + valStr +\
                srcDirListA[i][2:], temp.astype(np.float32))
    
    
for i in range(splitVal + splitTrain,splitVal + splitTrain + splitTest):
    try:
        #Assume B is colored, take out transpose if it isnt
        temp = cv2.imread(sourceB + srcDirListB[i])
        a = cv2.imwrite(sinkB + testStr +\
                    srcDirListB[i][2:], temp.astype(np.float32))
        if not a:
            print('Couldnt write image in A')
        
    except:
        print('Found a greyscale in OG')
        continue
    
    
    #Assume B is greyscale, take out flag if it isnt
    temp = cv2.imread(sourceA + srcDirListA[i], 0)
    cv2.imwrite(sinkA + testStr +\
                srcDirListA[i][2:], temp.astype(np.float32))

    
    
    


#tempSrc = cv2.imread(srcPaths[i]).transpose((2,0,1))


