from Aruco_signs import return_opponent_positions, setup_for_match
from Aruco_signs import *

import cv2 as cv

if __name__ == "__main__":
    ORTHO_PROJ, OPPONENT_IDS = setup_for_match("yellow")
    img = cv.imread('updated_pos/board_img_1.png')
    cv.imshow('t', img)
    cv.waitKey(0)
    positions = return_opponent_positions(img)
    print(positions)
