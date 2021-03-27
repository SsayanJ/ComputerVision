import numpy as np
import cv2 as cv

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def get_Contours(img):
    contours,hierarchy=cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv.contourArea(cnt)
        if area>500: # to avoid picking up noise (not the case on this image)
            cv.drawContours(img_contour,cnt,-1,(255,0,0),3)
            perimeter=cv.arcLength(cnt,True)
            approx=cv.approxPolyDP(cnt,0.02*perimeter,True)
            objCor=(len(approx))
            x,y, width, height=cv.boundingRect(approx)
            cv.rectangle(img_contour,(x,y),(x+width,y+height),(0,255,0),2)

            if objCor==3: object_type='Triangle'
            elif objCor==8: object_type='Circle'
            elif objCor==4: 
                aspect_ratio=width/height
                if aspect_ratio >0.95 and aspect_ratio<1.05: object_type='Square'
                else:object_type='Rectangle'
                    
            else: object_type='Not recognised'

                
            cv.putText(img_contour,object_type,(x+width//2-10,y+height//2-10),cv.FONT_HERSHEY_COMPLEX,3,(0,0,0),4)


# kernel = np.array([[0, -1, 0], [-1, 5, -1],[0, -1, 0]])
# create kernel
kernel = np.ones((15,15), np.uint8)
kernel2 = np.ones((7,7), np.uint8)

img2=cv.imread("Robot_signs/MorphX-Cup2.jpg")
img3=cv.imread("Robot_signs/Cup2.jpg")
img1=cv.imread("Robot_signs/Full_board1.PNG")
img=img3
imgHSV=cv.cvtColor(img,cv.COLOR_BGR2HSV)
imgHSV1=cv.erode(imgHSV, kernel)
imgHSV1_1=cv.erode(imgHSV1, kernel)
imgHSV1_2=cv.erode(imgHSV1_1, kernel)
imgHSV2=cv.dilate(imgHSV, kernel)
imgHSV3= cv.morphologyEx(imgHSV, cv.MORPH_GRADIENT, kernel)

lower_red=np.array([0,168,200])
lower_green=np.array([66,136,29])
upper_red=np.array([2,255,255])
upper_green=np.array([85,255,176])
test=False
if test:  # For red
    mask=cv.inRange(imgHSV1_2,lower_red,upper_red)
else: # For green
    mask=cv.inRange(imgHSV1_2,lower_green,upper_green)

result_Img=cv.bitwise_and(img,img,mask=mask)

img_contour=img.copy()
# img_blank=np.zeros_like(img)    
img_Grey=cv.cvtColor(result_Img,cv.COLOR_BGR2GRAY)
img_Blur=cv.GaussianBlur(img_Grey,(25,25),1)
img_Canny=cv.Canny(img_Blur,50,50)
get_Contours(img_Canny)

# 
# img_Stacked=stackImages(0.2,([imgHSV1_2,imgHSV1],[imgHSV3,imgHSV1_1]))
img_Stacked=stackImages(0.2,([img,img_Blur],[img_Canny,img_contour]))
# cv.imwrite("Robot_signs/Cup1.jpg",imgHSV3)
cv.imshow('All',img_Stacked)


cv.waitKey(0)