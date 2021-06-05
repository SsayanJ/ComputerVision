import cv2 as cv
import cv2.aruco as aruco
import json
import time
import numpy as np


def scale_pic(src, ratio=50):
    scale_percent = ratio

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv.resize(src, dsize)

    return output
########## GENERAL SETUP, TO BE KEPT FOR EACH OPTIONS:    ##############""

COLOR= (255,0,0)
aruco_signs_legend={1:"bleu",2:"bleu",3:"bleu",4:"bleu",5:"bleu",
                    6:"jaune", 7:'jaune', 8:'jaune', 9:'jaune', 10:'jaune',
                    42:'Board center'}
# Load ARUCO parameters
aruco_Dict=aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_Params=aruco.DetectorParameters_create()
# Recover fisheye camera settings from JSON config file
with open('Fisheye_config.json') as config:
    fisheye_cfg=json.load(config)

mtx=np.asarray(fisheye_cfg['mtx'])
dist=np.asarray(fisheye_cfg['dist'])

DIM= (1280,720)  # standard dimension of pics from camera
map1,map2=cv.fisheye.initUndistortRectifyMap(mtx,dist,np.eye(3),mtx,DIM,cv.CV_16SC2)

######## SINGLE IMAGE: Load image to check for single image mode, just uncomment the next 2 paragraphs to use on single image
# img=cv.imread('Robot_signs/Full_board1.PNG')
# img2=cv.imread('Robot_signs/Sign_first_angle.jpg')
# img3=cv.imread('Robot_signs/Sign_1000G_400H.jpg')
# img4=cv.imread('Robot_signs/Sign_500D_1000H_bad.jpg')
# img5=cv.imread('Robot_signs/Sign_500D_1000H.jpg')
# img6=cv.imread("board_img_1.png")
# img7=cv.imread("test_pics/board_img_1.png")
# 
# for i in range(50):
#     curr_img = cv.imread(f"test_pics/board_img_{i}.png")
# #     img2 = cv.imread(f"test_pics/prev/board_img_{i}.png")
#     curr_img = cv.remap(curr_img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
# #     img2 = cv.remap(img2, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
#     cv.imwrite(f'train/result_0{i}.jpg', curr_img)
#     cv.imwrite(f'train/result_1{i}.jpg', img2)
# 
# curr_img=img7
# curr_img=scale_pic(curr_img,50)

# Code to check on a single image
# curr_img = cv.remap(curr_img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
# corners, ids , rejected= aruco.detectMarkers(curr_img,aruco_Dict,parameters=aruco_Params)
# curr_img = cv.aruco.drawDetectedMarkers(curr_img, corners, ids)
# corners_42 = np.float32(corners[list(ids).index(42)])
# coord_42 = np.float32([[1450, 700],
#             [1550, 700],
#             [1550, 800],
#             [1450, 800]])
# init_Lsupport = np.float32([[286, 77],
#             [896, 79],
#             [1111, 445],
#             [272, 511]])
# 
# out_Lsupport = np.float32([[400, 900],
#             [1900, 900],
#             [2100, 1700],
#             [800, 1800]])
# init_points = np.float32([[269, 77],
#             [1168, 178],
#             [1089, 560],
#             [159, 438]])
# 
# out_points = np.float32([[400, 800],
#             [2500, 800],
#             [2100, 1700],
#             [600, 1700]])
# ortho_proj = cv.getPerspectiveTransform(corners_42, coord_42)
# ortho_proj = cv.getPerspectiveTransform(init_Lsupport, out_Lsupport)
# out = cv.warpPerspective(curr_img, ortho_proj,(3000, 2000),flags=cv.INTER_LINEAR)
# cv.imwrite('Warp_Lsupport.jpeg', out)
# cv.imwrite('Undistor_Lsupport.jpeg', curr_img)
# out = scale_pic(out, 40)
# for i,val in enumerate(corners):
#     try:
#         sign=aruco_signs_legend[int(ids[i])]
#         id_min=val[0].sum(axis=1).argmin()
#         id_max=val[0].sum(axis=1).argmax()
#         UL_x,UL_y=int(val[0][id_min][0]),int(val[0][id_min][1])
#         DR_x,DR_y=int(val[0][id_max][0]),int(val[0][id_max][1])
#         center_x,center_y=(DR_x+UL_x)//2,(DR_y+UL_y)//2
#         cv.rectangle(curr_img,(UL_x,UL_y),(DR_x,DR_y),COLOR,3)
#         cv.putText(curr_img,sign,(UL_x,UL_y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,COLOR,2)
#         cv.circle(curr_img,(center_x,center_y), 2, (0,255,255), -1)
#     except KeyError:
#         continue

# cv.imshow('Result',curr_img)
# cv.imshow('warp', out)
# cv.waitKey(0)



######## FROM WEBCAM STREAM 

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,DIM[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT,DIM[1])

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('POC_Signs_reco.avi',fourcc, 20.0, (640,480))

while True:
    
    # GET CAMERA IMAGE 
    success, img = cap.read()
    # DISPLAY THE DETECTED OBJECTS
    corners, ids , rejected= aruco.detectMarkers(img,aruco_Dict,parameters=aruco_Params)
    aruco.drawDetectedMarkers(img,corners)
    for i,val in enumerate(corners):
        try:
            curr_id=int(ids[i])
            if curr_id in aruco_signs_legend:
                sign=aruco_signs_legend[int(ids[i])]
            else:
                sign=str(curr_id)
            id_min=val[0].sum(axis=1).argmin()
            id_max=val[0].sum(axis=1).argmax()
            UL_x,UL_y=int(val[0][id_min][0]),int(val[0][id_min][1])
            DR_x,DR_y=int(val[0][id_max][0]),int(val[0][id_max][1])
            center_x,center_y=(DR_x+UL_x)//2,(DR_y+UL_y)//2
#             cv.rectangle(img,(UL_x,UL_y),(DR_x,DR_y),COLOR,2)
            cv.putText(img,sign,(UL_x,UL_y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,COLOR,2)
#             cv.circle(img,(center_x,center_y), 2, (0,255,255), -1)
        except KeyError:
            continue

    cv.imshow("Result", img)
    dst=cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
#     out.write(img)
    cv.imshow("Straightened image", dst)

    if cv.waitKey(1) & 0xFF == ord('q'):
         break


####### from sample video

# cap = cv.VideoCapture('CDF_sample.mp4')
# speed=25
# 
# while (cap.isOpened()):
#     
#     # GET CAMERA IMAGE 
#     success, img = cap.read()
#     
#     # DISPLAY THE DETECTED OBJECTS
#     corners, ids , rejected= aruco.detectMarkers(img,aruco_Dict,parameters=aruco_Params)
#     for i,val in enumerate(corners):
#         try: 
#             sign=aruco_signs_legend[int(ids[i])]
#             id_min=val[0].sum(axis=1).argmin()
#             id_max=val[0].sum(axis=1).argmax()
#             UL_x,UL_y=int(val[0][id_min][0]),int(val[0][id_min][1])
#             DR_x,DR_y=int(val[0][id_max][0]),int(val[0][id_max][1])
#             center_x,center_y=(DR_x+UL_x)//2,(DR_y+UL_y)//2
#             cv.rectangle(img,(UL_x,UL_y),(DR_x,DR_y),color,3)
#             cv.putText(img,sign,(UL_x,UL_y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
#             cv.circle(img,(center_x,center_y), 3, (0,255,255), -1)
#         except KeyError:
#             continue
# 
#             
# 
#     cv.imshow("Result", img)
# 
#     if cv.waitKey(speed) & 0xFF == ord('q'):
#          break
# cap.release()
# cv.destroyAllWindows()