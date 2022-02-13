#!/usr/bin/env python
# coding: utf-8

# ## SIFT Feature Matching
# 

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[3]:


# https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html


# In[48]:


# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
Img_1 = cv2.imread('./adobe_panoramas/data/hotel/hotel-00.png',cv2.IMREAD_GRAYSCALE)          
Img_2 = cv2.imread('./adobe_panoramas/data/hotel/hotel-01.png',cv2.IMREAD_GRAYSCALE) 
sift = cv2.SIFT_create()

orb = cv2.ORB_create()
Kp1, Des1 = orb.detectAndCompute(Img_1,None)
Kp2, Des2 = orb.detectAndCompute(Img_2,None)
SiftKp1, SiftDes1 = sift.detectAndCompute(Img_1,None)
SiftKp2, SiftDes2 = sift.detectAndCompute(Img_2,None)
bf_match = cv2.BFMatcher()
matches = bf_match.knnMatch(Des1,Des2,k=2)

List_good = []     # Empty List for the matches found in the two images of the hotel
for m,n in matches:
    if m.distance < 0.90*n.distance:
        List_good.append([m])


Result_Img = cv2.drawMatchesKnn(Img_1,Kp1,Img_1,SiftKp1,List_good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
Result_Img2 = cv2.drawMatchesKnn(Img_1,Kp1,Img_1,Kp2,List_good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# In[49]:


plt.figure(figsize=(20,20))
plt.imshow(Result_Img)
plt.title("Feature Matching for the Hotel Images")
plt.show()


# In[50]:


plt.figure(figsize=(20,20))
plt.imshow(Result_Img2)
plt.title("Feature Matching for the Hotel Images")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




