# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 00:18:13 2021

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

#Importing a custom library 
from transforms import RGBTransform

#------------------------OPENCV Assignment(Part-2)-----------------------------

#Reading Image
img = cv2.imread('CVA2.jpg')
cv2.imshow('Original',img)

#Resize without distortion
resize_w = 1200 / img.shape[1]
dim = (1200, int(img.shape[0] * resize_w))

#Resizing
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)

"""--------------------------RED PATCH CORRECTION-----------------------------"""

#Plotting to find ROI coordinates
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

#Mask of the shape of the image and red patch area extraction
img1 = cv2.imread('CVA2.jpg')
mask = np.zeros(img1.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 0), (300, 300), 255, -1)
cv2.imshow("Rectangular Mask", mask)

#Cutting the mask out from image
masked = cv2.bitwise_and(img1, img1, mask=mask)
cv2.imshow("Mask Applied to Image", masked)

"""APPROACH: Two more masks, one beside and one below the main mask (red patch) 
were chosen,split and their mean R,G,B values were computed. (/CODE ATTACHED BELOW/)
A new layer comprising of the average colour information from neighbours was chosen and applied on
and RGBTransform function was applied on the main mask (red patch)"""

#Coverting to the mask area to grayscale to remove red tint
masked_grayscale = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
cv2.imshow('Mask Grayscale',masked_grayscale)

#Back to RGB to perform RGBTransform (custom function)
backtorgb_masked = cv2.cvtColor(masked_grayscale,cv2.COLOR_GRAY2RGB)
cv2.imshow('Grayscale2RGB Mask',backtorgb_masked)
#Applying RGB Transform---------------------
backtorgb_masked_img = Image.fromarray(np.uint8(backtorgb_masked),'RGB')
backtorgb_masked_img.getbands()
#RGB Values from neighbouring tiles
masked_transform = RGBTransform().mix_with((60,52,36),factor=0.12).applied_to(backtorgb_masked_img)
masked_transform.show()

#Converting back to BGR to merge with main image
masked_transform_arr = np.uint8(masked_transform)
masked_transform_bgr = cv2.cvtColor(masked_transform_arr, cv2.COLOR_RGB2BGR)
cv2.imshow('RGB2BGR Mask',masked_transform_bgr)

#Adding brightness
a = 4
b = 2
masked_transform_bgrbright = cv2.convertScaleAbs(masked_transform_bgr, alpha=a, beta=b)
cv2.imshow("Brightness Adjusted",masked_transform_bgrbright)

#Merging back into main image
img_new = img
for i in range(0,300):
    for j in range(0,300):
        img_new[0:i,0:j,0] = masked_transform_bgrbright[0:i,0:j,0]
        img_new[0:i,0:j,1] = masked_transform_bgrbright[0:i,0:j,1]
        img_new[0:i,0:j,2] = masked_transform_bgrbright[0:i,0:j,2]
cv2.imshow("img_new",img_new)
cv2.imwrite("Redpatch Alteration.jpg",img_new)

"""--------------------------BRIGHTNESS CORRECTION-----------------------------"""

#Adding brightness
a = 1
b = 12
img_new_bright = cv2.convertScaleAbs(img_new, alpha=a, beta=b)
cv2.imshow("Brightness Adjusted",img_new_bright)
cv2.imwrite("Brightness Alteration.jpg",img_new_bright)


"""--------------------------EDGE DETECTION -----------------------------"""

#Canny edge was used to delineate edges of the cars body 
img = cv2.imread("E://IPWP_SS21//Class_Files//A2.jpg",0)
Canny_edge = cv2.Canny(img, 140, 140)
cv2.imshow('Canny edge method', Canny_edge)
cv2.imwrite("Carbody Edge Extraction.jpg",Canny_edge)

#-------------------------------END---------------------------------------#
.
.
.
.
.
.
.
#-------------------------------REFERENCE CODE---------------------------------------#

#--------------------------------------Neighbour Mask1 RIGHT--------------------
img1 = cv2.imread('CVA2.jpg')
mask_r = np.zeros(img1.shape[:2], dtype="uint8")
cv2.rectangle(mask_r, (300, 0), (600, 300), 255, -1)
cv2.imshow("Rectangular Maskr", mask_r)
#Apply R_Mask
right_masked = cv2.bitwise_and(img1, img1, mask=mask_r)
cv2.imshow("Mask Applied to Image", right_masked)
#Splitting channel 
(B_r, G_r, R_r) = cv2.split(right_masked)
cv2.imshow("Red", R_r)
cv2.imshow("Green", G_r)
cv2.imshow("Blue", B_r)
#Red Mean
R_r_arr_s = R_r[0:300,300:600]
R_r_arr =  np.uint8(R_r_arr_s)
R_r_mean = np.mean(R_r_arr)
#Green Mean
G_r_arr_s = G_r[0:300,300:600]
G_r_arr =  np.uint8(G_r_arr_s)
G_r_mean = np.mean(G_r_arr)
#Blue Mean
B_r_arr_s = B_r[0:300,300:600]
B_r_arr =  np.uint8(B_r_arr_s)
B_r_mean = np.mean(B_r_arr)
#---------------------------------------------------------------------



#-------------------------------------Neighbour Mask1 BOTTOM---------------------
img1 = cv2.imread('CVA2.jpg')
mask_b = np.zeros(img1.shape[:2], dtype="uint8")
cv2.rectangle(mask_b, (0, 300), (300, 600), 255, -1)
cv2.imshow("Rectangular Maskb", mask_b)
#Apply B_Mask
bottom_masked = cv2.bitwise_and(img1, img1, mask=mask_b)
cv2.imshow("Mask Applied to Image", bottom_masked)
#Splitting channel 
(B_b, G_b, R_b) = cv2.split(bottom_masked)
cv2.imshow("Red", R_b)
cv2.imshow("Green", G_b)
cv2.imshow("Blue", B_b)
#Red Mean
R_b_arr_s = R_b[300:600,0:300]
R_b_arr =  np.uint8(R_b_arr_s)
R_b_mean = np.mean(R_b_arr)
#Green Mean
G_b_arr_s = G_b[300:600,0:300]
G_b_arr =  np.uint8(G_b_arr_s)
G_b_mean = np.mean(G_b_arr)
#Blue Mean
B_b_arr_s = B_b[300:600,0:300]
B_b_arr =  np.uint8(B_b_arr_s)
B_b_mean = np.mean(B_b_arr)
-----------------------------------------------------------------------

"""The average values were chosen as follows 
RED channel value from neighbour tiles = 60
GREEN value from neighbour tiles = 52
BLUE value from neighbour tiles = 36"""  

















