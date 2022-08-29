# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:26:57 2021

@author: Rounak Goel
"""

import numpy as np
import scipy as sp
import pandas as pd
import cv2 as cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

os.chdir("E://IPWP_SS21//Class_Files")

#------------------------OPENCV Assignment(Part-1)-----------------------------

img = cv2.imread('A2.jpg')
cv2.imshow('A2',img)

#----------------1.Comments about the image-----------
"""
1. The image is very sharp, with clearly distinguishable foreground, subject and background
2. The contrast on the image is very good with clearly distinguishable features and 
   colour breaks espeacially around the subject,
3. The pattern on the rightside wall has poor contrast and is underexposed,  
   little exposure correction/sharpness around the background may be advised
4. The building in the background seem to be overexposed and excessively white 
5. Reflections from the lightsources in the image create a little noise (blue above the headlights,raindrops)  
   and imperfections (subjective) in the image.
   
"""
#----------------2. Histogram Evaluation-------------------
#Grayscale Histogram
split_img = cv2.split(img)
histSize = 256
histRange = (0, 256)
pixel_values = cv2.calcHist(split_img,[0],None,[histSize],histRange)
plt.plot(pixel_values, label = 'pixel_values',color = 'black')
plt.legend()

#RGB Histogram
r = cv2.calcHist(split_img,[0],None,[histSize],histRange)
g = cv2.calcHist(split_img,[1],None,[histSize],histRange)
b = cv2.calcHist(split_img,[2],None,[histSize],histRange)
plt.plot(r, label = 'r',color = 'red')
plt.plot(g, label = 'g',color = 'green')
plt.plot(b, label = 'b', color = 'blue')
plt.legend()

"""-------------Make comments based on the distribtuion of r,g,b channels.
1.The histogram shows significant spike in the dark/shadows region and 
and is dominated by greens and reds
2.The histogram is well distributed and hence has a good tonal range
3.The shadows to midtones transition region comprises lot of information attributing 
to a fine image
4.The histogram dips along the midtones and gains slighly (specifically greens) 
  along the highlights
5.The whites/highlights are dominated by the blue channel

"""
#----------------2a. Histogram Evaluation-------------------

"""------------Do we need to correct the histogram?
As mentioned the tonal range of the image is good and is not narrow 
hence an equalization or distribution of intensity across the image depth is not required
*An equalization or maybe preferred along certain areas of the image (not for entire) to 
distribute the highs and lows better
"""
#----------------HISTOGRAM EQUALIZATION (CLAHE)----------------

img = cv2.imread("E://IPWP_SS21//Class_Files//A2.jpg")
img = cv2.resize(img, dsize = (1200,800))
cv2.imshow('Unequalize',img)
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
cv2.imshow('LAB',lab)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
cv2.imshow('Equalized',img_rgb)

"""------------Justification why we may not need it

The colours on the subjects are little washed out, although few other
portions of the image have improved contrast but the subject in focus suffers from 
the equalization.
The rain drops become more significant and the blue colour bleeds are higher
hence it may not be required.

"""





