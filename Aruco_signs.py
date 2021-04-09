import cv2 as cv
import cv2.aruco as aruco

import time


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

color= (255,0,255)
aruco_signs_legend={2:"Opponent", 6:"Evol-PMI", 7:'Evol-PAL',42:'Board center'}
# Load ARUCO parameters
aruco_Dict=aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_Params=aruco.DetectorParameters_create()


######## SINGLE IMAGE: Load image to check for single image mode, just uncomment the next 2 paragraphs to use on single image
# img=cv.imread('Robot_signs/Full_board1.PNG')
# img2=cv.imread('Robot_signs/Sign_first_angle.jpg')
# img3=cv.imread('Robot_signs/Sign_1000G_400H.jpg')
# img4=cv.imread('Robot_signs/Sign_500D_1000H_bad.jpg')
# img5=cv.imread('Robot_signs/Sign_500D_1000H.jpg')
# img6=cv.imread("Robot_signs/Test_WA.png")


# curr_img=img6
# curr_img=scale_pic(curr_img,50)

# # Code to check on a single image
# corners, ids , rejected= aruco.detectMarkers(curr_img,aruco_Dict,parameters=aruco_Params)
# for i,val in enumerate(corners):
#     try:
#         sign=aruco_signs_legend[int(ids[i])]
#         id_min=val[0].sum(axis=1).argmin()
#         id_max=val[0].sum(axis=1).argmax()
#         UL_x,UL_y=int(val[0][id_min][0]),int(val[0][id_min][1])
#         DR_x,DR_y=int(val[0][id_max][0]),int(val[0][id_max][1])
#         center_x,center_y=(DR_x+UL_x)//2,(DR_y+UL_y)//2
#         cv.rectangle(curr_img,(UL_x,UL_y),(DR_x,DR_y),color,3)
#         cv.putText(curr_img,sign,(UL_x,UL_y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
#         cv.circle(curr_img,(center_x,center_y), 2, (0,255,255), -1)
#     except KeyError:
#         continue

# cv.imshow('Result',curr_img)

# cv.waitKey(0)



######## FROM WEBCAM STREAM 

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)

# Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('POC_Signs_reco.avi',fourcc, 20.0, (640,480))

while True:
    
    # GET CAMERA IMAGE 
    success, img = cap.read()
    # DISPLAY THE DETECTED OBJECTS
    corners, ids , rejected= aruco.detectMarkers(img,aruco_Dict,parameters=aruco_Params)
    for i,val in enumerate(corners):
        try: 
            sign=aruco_signs_legend[int(ids[i])]
            id_min=val[0].sum(axis=1).argmin()
            id_max=val[0].sum(axis=1).argmax()
            UL_x,UL_y=int(val[0][id_min][0]),int(val[0][id_min][1])
            DR_x,DR_y=int(val[0][id_max][0]),int(val[0][id_max][1])
            center_x,center_y=(DR_x+UL_x)//2,(DR_y+UL_y)//2
            cv.rectangle(img,(UL_x,UL_y),(DR_x,DR_y),color,2)
            cv.putText(img,sign,(UL_x,UL_y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            cv.circle(img,(center_x,center_y), 2, (0,255,255), -1)
        except KeyError:
            continue

    cv.imshow("Result", img)
#     out.write(img)

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