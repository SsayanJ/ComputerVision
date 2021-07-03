from Aruco_signs import return_opponent_positions, setup_for_match
from Aruco_signs import *

import cv2 as cv

if __name__ == "__main__":
    ortho_proj, opponent_ids = setup_for_match("yellow")
    # img = cv.imread('updated_pos/board_img_1.png')
    # img = cv.imread('angle_selection/trait_4.png')
    img = cv.imread('small_yellow/verres_config1_10.png')
    cv.imshow('t', img)
    cv.waitKey(0)
    girouette_color = girouette(img, "yellow")
    print(girouette_color)
    # opp_ids, positions = return_opponent_positions(
    #     img, ortho_proj, opponent_ids)
    # print(positions)
    # if len(positions) > 0:
    #     proj_positions = proj_pos(positions, BALISE_HEIGHT)
    #     print("corrected positions", proj_positions)
    # record_match(5)
