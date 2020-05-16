#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:12:15 2020

@author: ershov

"""



import torch
assert '.'.join(torch.__version__.split('.')[:2]) == '1.5'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset

import torchvision.transforms as transforms
import torch.nn.functional as F



from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import scipy
import os
import random






#Define a convNEt of desired architecture
def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?
    
    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None

    #torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
    #stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

    #conv 1 --> N x ch1 x H1 x W1 - conv 2 --> N x ch2 x H2 x W2 - Affine --> N x C
    #Thus, fc_w should be Hid x C, where Hid = ch2 * H2 * W2
    N,C,H,W = x.shape
    ch1,C,KH1,KW1 = conv_w1.shape
    ch2,ch1,KH2,KW2 = conv_w2.shape
    A,clasNum = fc_w.shape


    xN = F.conv2d(x, conv_w1, conv_b1, padding = 2)
    xN = F.relu(xN)
    xN = F.conv2d(xN, conv_w2, conv_b2, padding = 1)
    xN = F.relu(xN)
 
    scores = flatten(xN).mm(fc_w) + fc_b 

    return scores



#Test the afore-created convNet
def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]




#Create a random initialized tensor for weights
def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

#Create a 0 initialized tensor (for biases)
def zero_weight(shape):
    return torch.zeros(shape, dtype=dtype, requires_grad=True)





#Training helper function
def train_part2(model_fn, params, learning_rate, trainNum):
    """
    Train a model on CIFAR-10.
    
    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD
    - trainNum = number of iteration desires for training
    
    Returns: Nothing
    """
    for i in range(trainNum):
        # Move the data to the proper device (GPU or CPU)
#        x = x.to(device=device, dtype=dtype)
#        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()
            
            
            
#Accuracy checking helper function
def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.
    
    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model
    
    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
#            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
#            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))




#myDataLoader Class to retrieve mini batches
class myDataLoader:
  def __init__(self, imgSrcDest, imgSinkDest):
    self.imgSrc = imgSrcDest
    self.imgSink = imgSinkDest
    
    self.srcList = os.listdir(self.imgSrc)
    self.srcList.remove('.DS_Store')
    
    self.sinkList = os.listdir(self.imgSink)
    self.sinkList.remove('.DS_Store')
    
  def getRandImg(self, fromSink = False):
    if not fromSink:
        return cv2.imread(self.imgSrc + random.choice(self.srcList))
    else:
        return cv2.imread(self.imgSink + random.choice(self.sinkList))
    

  def getMiniBatch(self, batchSize):
    self.srcMbatchNames = random.sample(self.srcList, k = batchSize)
    self.sinkMbatchNames = random.sample(self.sinkList, k = batchSize)
    
    #Create a matrix containing paths for all of the images in the minibatch
    srcPaths = [self.imgSrc + i for i in self.srcMbatchNames]
    sinkPaths = [self.imgSink + i for i in self.sinkMbatchNames]
    
    self.srcMbatch = batchSize*[0]
    self.sinkMbatch = batchSize*[0]
    
    for i in range(batchSize):
        #Read from files
        #Transpose so that dimensions are CxHxW (confirmed)
        #All images have different dimensions
        tempSrc = cv2.imread(srcPaths[i]).transpose((2,0,1))
        tempSink = cv2.imread(sinkPaths[i]).transpose((2,0,1))
        
        #Transform to torch tensor
        self.srcMbatch[i] = torch.from_numpy(tempSrc)
        self.sinkMbatch[i] = torch.from_numpy(tempSink)
        
        
    


        


###ALL functions end here #####
            
            
            
            
#Starter initializing variables
dtype = torch.float32 #defines the default output datatype in helper functions
print_every = 100 #defines how often we print in helper functions
file_og = 'training_images/Original/'
file_sk = 'training_images/Sketch/'
file_bw = 'training_images/Grayscale/'
fext = '.jpg'

thisData = myDataLoader(file_sk, file_bw)
thatData = myDataLoader(file_bw, file_og)

thisData.getMiniBatch(64)
print(thisData.srcMbatch)

##TODO
#finish up the helper functions to be compatible with our workflow
#   -mainly, make the dataloader thing a class that takes images in my folders and generates and new mini batch when called (which is what we do currently too)
#   -build a very general convolutional net architecture
#   -Decide on a loss functiin
#   -Try to train some shit!






#coco_val = dset.COCO('toy_temp/', train=True, download=True,
#                             transform=transform)


cap = dset.CocoCaptions(root = 'coco/images/val2017',
                        annFile = 'coco/annotations/captions_val2017.json',
        
                        transform=transforms.ToTensor())
#f = torch.from_numpy(c)

#loader_train = DataLoader(cifar10_train, batch_size=64, 
#                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
#
#cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
#                           transform=transform)
#loader_val = DataLoader(cifar10_val, batch_size=64, 
#                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
#
#cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, 
#                            transform=transform)
#loader_test = DataLoader(cifar10_test, batch_size=64)





#
#
##Reference code:
#learning_rate = 3e-3
#
#channel_1 = 32
#channel_2 = 16
#
#conv_w1 = None
#conv_b1 = None
#conv_w2 = None
#conv_b2 = None
#fc_w = None
#fc_b = None
#
#################################################################################
## TODO: Initialize the parameters of a three-layer ConvNet.                    #
#################################################################################
## *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#conv_w1 = random_weight((channel_1,3,5,5)) #[ch1,C,fH1,fW1]
#conv_b1 = zero_weight((channel_1)) #[ch1]
#conv_w2 = random_weight((channel_2,channel_1,3,3)) #[ch2,ch1,fH2,fW2]
#conv_b2 = zero_weight((channel_2)) #[ch2]
# 
##now calculate size of H_ and W_ after 2 convnets
#zP1 = 2
#zP2 = 1
#H_ = (32 + 2*zP1 - 5 + 1) + 2*zP2 - 3 + 1
#fc_w = random_weight((channel_2*H_*H_, 10)) #[ch2*fH2*fW2,classNum]
#fc_b = zero_weight((10)) #[classNum]
#
#
##3 layer net params is expecting: conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
#funcParam = [w1,b1,w2,b2,w3,b3]
#
#
#pass
#
## *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#################################################################################
##                                 END OF YOUR CODE                             #
#################################################################################
#
#params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
#train_part2(three_layer_convnet, params, learning_rate)
