#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# OpenCV has a function, cv.goodFeaturesToTrack(). It finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if you specify it). As usual, image should be a grayscale image. Then you specify number of corners you want to find. Then you specify the quality level, which is a value between 0-1, which denotes the minimum quality of corner below which everyone is rejected. Then we provide the minimum euclidean distance between corners detected.

# In[9]:


#https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
img = cv2.imread('./adobe_panoramas/data/office/office-03.png')
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
det_corners = cv2.goodFeaturesToTrack(gray_image,10,0.02,10)
det_corners = np.int0(det_corners)
plt.imshow(img)
for i in corners:
    x,y = i.ravel()
    plt.scatter(x, y, 50, c="r", marker="x")
plt.show()


# In[ ]:




