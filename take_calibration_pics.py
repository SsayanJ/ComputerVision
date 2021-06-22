import cv2

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

cv2.namedWindow("test")

img_counter = 0
timer=0

while img_counter<50:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1                                       )
    timer+=1
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif timer%50 == 0:
        # Every 250 frames
        img_name = "updated_pos/board_img_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()