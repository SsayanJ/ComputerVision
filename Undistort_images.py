import numpy as np
import cv2 as cv
import glob
import json

# Recover fisheye camera settings from JSON config file
with open('Fisheye_config.json') as config:
    fisheye_cfg=json.load(config)

mtx=fisheye_cfg['mtx']
dist=fisheye_cfg['dist']

mtx=np.asarray(mtx)
dist=np.asarray(dist)
DIM= (1280,720)  # standard dimension of pics from camera

img = cv.imread('board_img_1.png')

map1,map2=cv.fisheye.initUndistortRectifyMap(mtx,dist,np.eye(3),mtx,DIM,cv.CV_16SC2)
dst=cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

cv.imwrite('calibresult3.png', dst)


# Change balance
# balance=1.
# img = cv.imread('calibresult2.png')
# img_dim = img.shape[:2][::-1]  
# 
# K = np.zeros((3, 3))
# D = np.zeros((4, 1))
# 
# scaled_K = K * img_dim[0] / DIM[0]  
# scaled_K[2][2] = 1.0  
# new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
#     img_dim, np.eye(3), balance=balance)
# map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3),
#     new_K, img_dim, cv.CV_16SC2)
# undist_image = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR,
#     borderMode=cv.BORDER_CONSTANT)
# 
# cv.imwrite('balance_test.png', undist_image)

