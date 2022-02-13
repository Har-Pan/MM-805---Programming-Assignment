#!/usr/bin/env python
# coding: utf-8

# ## SIFT Feature Detection
# 

# Implementing the SIFT also known as Scale Invariant Feature Transform feature detection.
# 
# In 2004, D.Lowe, University of British Columbia, came up with a new algorithm, Scale Invariant Feature Transform (SIFT) in his paper, Distinctive Image Features from Scale-Invariant Keypoints, which extract keypoints and compute its descriptors.
# 
# There are mainly four steps involved in SIFT algorithm:
# 
# 1. Scale-space Extrema Detection
# 2. Keypoint Localization
# 3. Orientation Assignment
# 4. Keypoint Descriptor
# 5. Keypoint Matching

# In[1]:


#https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
import cv2 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:




def detect_feature(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    feature_extractor = cv2.SIFT_create()
    p, desc = feature_extractor.detectAndCompute(gray_image, None)
    result_Img = cv2.drawKeypoints(rgb_image, p, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return result_Img


# In[18]:


rgb = cv2.cvtColor(cv2.imread('./adobe_panoramas/data/hotel/hotel-00.png'), cv2.COLOR_BGR2RGB)
Image_detected_nor = detect_feature(rgb)

figsize = (9, 9)
plt.figure(figsize=figsize)
plt.imshow(Image_detected_nor)
plt.title("Keypoints_detected")
plt.show()


# In[29]:


rgb_rotated = cv2.rotate(rgb, cv2.cv2.ROTATE_90_CLOCKWISE)
Image_detected_rot = detect_feature(rgb_rotated)

figsize = (9, 9)
plt.figure(figsize=figsize)
plt.imshow(detected_image_rotated)
plt.title("Keypoints_detected")
plt.show()


# In[24]:


def resize_img(image,scale_percent):
    
    width_image = int(image.shape[1] * scale_percent / 100)
    height_image = int(image.shape[0] * scale_percent / 100)
    dimension_image = (width_image, height_image)

    resized_image = cv2.resize(image, dimension_image, interpolation = cv2.INTER_AREA)
    return resized_image


# In[26]:


rgb_scaled = resize_img(rgb,150)
Image_after_scale = detect_feature(rgb_scaled)

figsize = (9, 9)
plt.figure(figsize=figsize)
plt.imshow(Image_after_scale)
plt.title("Keypoints_detected")
plt.show()


# # Final Results

# In[28]:


plt.figure(figsize=(20,20))
plt.subplot(131)
plt.imshow(Image_detected_nor)
plt.subplot(132)
plt.imshow(Image_detected_rot)
plt.subplot(133)
plt.imshow(Image_after_scale)


# In[ ]:




