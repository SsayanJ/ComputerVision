import numpy as np
import cv2 as cv
import glob
import json


def jsonify(data):
    json_data = dict()
    for key, value in data.items():
        if isinstance(value, list): # for lists
            value = [ jsonify(item) if isinstance(item, dict) else item for item in value ]
        if isinstance(value, dict): # for nested lists
            value = jsonify(value)
        if isinstance(key, int): # if key is integer: > to string
            key = str(key)
        if type(value).__module__=='numpy': # if value is numpy.*: > to python list
            value = value.tolist()
        json_data[key] = value
    return json_data

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1,7*7,3), np.float32)
objp[0,:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
fisheye_cfg={}
images = glob.glob('ongoing_calib/*.png')
counter=0
accepted=0
used_imgs=[]
for fname in images:
    print('checking image' , counter)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        accepted+=1
        used_imgs.append(counter)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,7), corners2, ret)
        cv.imshow('img'+str(counter), img)
        cv.waitKey(200)
        cv.destroyAllWindows()
    counter+=1    
print(f'Number of Image accepted {accepted}/{counter}')
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.fisheye.calibrate(objpoints,
                                                  imgpoints, gray.shape[::-1], None, None)
img = cv.imread('ongoing_calib/board_img_29.png')
#  Save the camera matrices in config
fisheye_cfg['mtx']=mtx
fisheye_cfg['dist']=dist
# undistort a sample image
map1,map2=cv.fisheye.initUndistortRectifyMap(mtx,dist,np.eye(3),mtx,gray.shape[::-1],cv.CV_16SC2)
# fisheye_cfg['map1']=map1
# fisheye_cfg['map2']=map2

cfg = jsonify(fisheye_cfg)
with open('Fisheye_config.json', 'w') as config:
    json.dump(cfg, config)

dst=cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

cv.imwrite('calibresult.png', dst)