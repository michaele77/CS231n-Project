#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:15:47 2020

@author: ershov
"""

from wand.image import Image
from wand.display import display

import subprocess
import numpy as np
import cv2
import scipy.ndimage

imPath = "PhotoSketch-master/examples/im_junnan.png"



imgCV = cv2.imread(imPath, cv2.IMREAD_COLOR)

#with Image(filename=imPath) as img:
#    print(type(img))
cv2.imwrite("out_image/imComp_original.jpg", imgCV.astype(np.float32))

#Adding a smoothing filter
#    smoothOriginal_5 = scipy.ndimage.gaussian_filter(img, sigma=5)
#    cv2.imwrite("out_image/imComp_smooth_original_s5.jpg", np.float32(smoothOriginal_5))
#    smoothOriginal_10 = scipy.ndimage.gaussian_filter(img, sigma=10)
#    cv2.imwrite("out_image/imComp_smooth_original_s10.jpg", np.float32(smoothOriginal_10))
#    smoothOriginal_30 = scipy.ndimage.gaussian_filter(img, sigma=30)
#    cv2.imwrite("out_image/imComp_smooth_original_s30.jpg",np.float32(smoothOriginal_30))

#Now do the sketch transform
print(imgCV.shape)
sketch_gray, sketch_color = cv2.pencilSketch(imgCV, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
cv2.imwrite("out_image/sktchComp_0.jpg", sketch_gray)
ogSketch = sketch_gray
sketch_gray, sketch_color = cv2.pencilSketch(imgCV, sigma_s=60, sigma_r=0.15, shade_factor=0.05)
sketch_1 = sketch_gray
cv2.imwrite("out_image/sktchComp_1.jpg", sketch_gray)
sketch_gray, sketch_color = cv2.pencilSketch(imgCV, sigma_s=60, sigma_r=0.2, shade_factor=0.05)
goodSketch = sketch_gray #I like the output of this one
cv2.imwrite("out_image/sktchComp_2.jpg", sketch_gray)
sketch_gray, sketch_color = cv2.pencilSketch(imgCV, sigma_s=60, sigma_r=0.25, shade_factor=0.05)
cv2.imwrite("out_image/sktchComp_3.jpg", sketch_gray)
sketch_gray, sketch_color = cv2.pencilSketch(imgCV, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
cv2.imwrite("out_image/sktchComp_4.jpg", sketch_gray)
sketch_gray, sketch_color = cv2.pencilSketch(imgCV, sigma_s=60, sigma_r=0.07, shade_factor=0.15)
cv2.imwrite("out_image/sktchComp_5.jpg", sketch_gray)
sketch_gray, sketch_color = cv2.pencilSketch(imgCV, sigma_s=60, sigma_r=0.07, shade_factor=0.2)
cv2.imwrite("out_image/sktchComp_6.jpg", sketch_gray)
sketch_gray, sketch_color = cv2.pencilSketch(imgCV, sigma_s=60, sigma_r=0.07, shade_factor=0.25)
cv2.imwrite("out_image/sktchComp_7.jpg", sketch_gray)

goodSketch_smth = scipy.ndimage.gaussian_filter(goodSketch, sigma=5)
cv2.imwrite("out_image/imComp_2_smooth.jpg", goodSketch_smth.astype(np.float32))



#Try dilation
kernel = np.ones((5, 5), np.float32)
goodSketch_dilate = cv2.dilate(ogSketch, kernel, iterations = 1)
cv2.imwrite("out_image/imComp_0_dilate.jpg", goodSketch_dilate.astype(np.float32))

kernel = np.ones((5, 5), np.float32)
goodSketch_dilate = cv2.dilate(sketch_1, kernel, iterations = 1)
cv2.imwrite("out_image/imComp_1_dilate.jpg", goodSketch_dilate.astype(np.float32))


kernel = np.ones((5, 5), np.float32)
goodSketch_dilate = cv2.dilate(goodSketch, kernel, iterations = 1)
cv2.imwrite("out_image/imComp_2_dilate.jpg", goodSketch_dilate.astype(np.float32))


#Try to erode the image
kernel = np.ones((5, 5), np.float32)
goodSketch_erode = cv2.erode(goodSketch, kernel, iterations = 1)
cv2.imwrite("out_image/imComp_2_erode.jpg", goodSketch_erode.astype(np.float32))

#Try median blur
goodSketch_medBlur = cv2.medianBlur(goodSketch, 3)
cv2.imwrite("out_image/imComp_2_medBlur.jpg", goodSketch_medBlur.astype(np.float32))

#Try gaussian denoising
#cv2.fastNlMeansDenoising(goodSketch, goodSketch_denoise)
goodSketch_denoise = cv2.fastNlMeansDenoising(goodSketch, 30.0, 7, 21)
cv2.imwrite("out_image/imComp_2_denoise.jpg", goodSketch_denoise.astype(np.float32))


sk1_denoise = cv2.fastNlMeansDenoising(sketch_1, 30.0, 7, 21)
cv2.imwrite("out_image/imComp_1_denoise.jpg", sk1_denoise.astype(np.float32))


#Now sharpen the good sketch 2 that was denoised
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
im_sharp = cv2.filter2D(goodSketch_denoise, -1, kernel)
cv2.imwrite("out_image/imComp_2_sharp.jpg", im_sharp.astype(np.float32))


#Conclusion: difference between sketch 2 and all of the blurring and denoising and shit didnt matter much
#Try to make the original pencil_sketch parameters match better
#Cant do much better than that




            
#            
#class Cartoonizer: 
#    """Cartoonizer effect 
#        A class that applies a cartoon effect to an image. 
#        The class uses a bilateral filter and adaptive thresholding to create 
#        a cartoon effect. 
#    """
#    def __init__(self): 
#        pass
#  
#    def render(self, img_rgb): 
#        img_rgb = cv2.imread(img_rgb) 
##        img_rgb = cv2.resize(img_rgb, (1366,768)) 
#        numDownSamples = 2       # number of downscaling steps 
#        numBilateralFilters = 50  # number of bilateral filtering steps 
#  
#        # -- STEP 1 -- 
#  
#        # downsample image using Gaussian pyramid 
#        img_color = img_rgb 
#        for _ in range(numDownSamples): 
#            img_color = cv2.pyrDown(img_color) 
#  
#        #cv2.imshow("downcolor",img_color) 
#        #cv2.waitKey(0) 
#        # repeatedly apply small bilateral filter instead of applying 
#        # one large filter 
#        for _ in range(numBilateralFilters): 
#            img_color = cv2.bilateralFilter(img_color, 9, 9, 7) 
#  
#        #cv2.imshow("bilateral filter",img_color) 
#        #cv2.waitKey(0) 
#        # upsample image to original size 
#        for _ in range(numDownSamples): 
#            img_color = cv2.pyrUp(img_color) 
#        #cv2.imshow("upscaling",img_color) 
#        #cv2.waitKey(0) 
#  
#        # -- STEPS 2 and 3 -- 
#        # convert to grayscale and apply median blur 
#        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) 
#        img_blur = cv2.medianBlur(img_gray, 3) 
#        #cv2.imshow("grayscale+median blur",img_color) 
#        #cv2.waitKey(0) 
#  
#        # -- STEP 4 -- 
#        # detect and enhance edges 
#        img_edge = cv2.adaptiveThreshold(img_blur, 255, 
#                                         cv2.ADAPTIVE_THRESH_MEAN_C, 
#                                         cv2.THRESH_BINARY, 9, 2) 
#        #cv2.imshow("edge",img_edge) 
#        #cv2.waitKey(0) 
#  
#        # -- STEP 5 -- 
#        # convert back to color so that it can be bit-ANDed with color image 
#        (x,y,z) = img_color.shape 
#        img_edge = cv2.resize(img_edge,(y,x))  
#        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB) 
#        cv2.imwrite("out_image/cv2_long_code_pencilSketch.png",img_edge) 
#        #cv2.imshow("step 5", img_edge) 
#        #cv2.waitKey(0) 
#        #img_edge = cv2.resize(img_edge,(i for i in img_color.shape[:2])) 
#        #print img_edge.shape, img_color.shape 
#        return cv2.bitwise_and(img_color, img_edge) 
#  
#tmp_canvas = Cartoonizer() 
#  
#file_name = imPath#File_name will come here 
#res = tmp_canvas.render(file_name) 
#  
#cv2.imwrite("out_image/Cartoon_version-dontknow.jpg", res) 
#









  
#img = cv2.imread(imPath) 
#   
## Edges 
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#gray = cv2.medianBlur(gray, 5) 
#edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,  
#                                         cv2.THRESH_BINARY, 9, 9) 
#   
## Cartoonization 
#color = cv2.bilateralFilter(img, 9, 250, 250) 
#cartoon = cv2.bitwise_and(color, color, mask=edges) 
#   
#   
#cv2.imshow("Image", img) 
#cv2.imshow("edges", edges) 
#cv2.imshow("Cartoon", cartoon) 
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 

#
#cmd = ["convert", "-monochrome", "-compress", "lzw", imPath, "tif:-"]
#fconvert = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#stdout, stderr = fconvert.communicate()
#assert fconvert.returncode == 0, stderr
#
## now stdout is TIF image. let's load it with OpenCV
#filebytes = np.asarray(bytearray(stdout), dtype=np.uint8)
#image = cv2.imdecode(filebytes, cv2.IMREAD_GRAYSCALE)

