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

img = cv.imread('ongoing_calib/board_img_22.png')

map1,map2=cv.fisheye.initUndistortRectifyMap(mtx,dist,np.eye(3),mtx,DIM,cv.CV_16SC2)
dst=cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

cv.imwrite('calibresult2.png', dst)