from Aruco_signs import return_opponent_positions, setup_for_match
from Aruco_signs import *

import cv2 as cv

if __name__ == "__main__":
    ortho_proj, opponent_ids = setup_for_match("blue")
#     # img = cv.imread('updated_pos/board_img_1.png')
#     # img = cv.imread('angle_selection/trait_4.png')
#     img = cv.imread('small_yellow/verres_config1_10.png')
#     cv.imshow('t', img)
#     cv.waitKey(0)
#     girouette_color = girouette(img, "yellow")
#     print(girouette_color)
#     # opp_ids, positions = return_opponent_positions(
#     #     img, ortho_proj, opponent_ids)
#     # print(positions)
#     # if len(positions) > 0:
#     #     proj_positions = proj_pos(positions, BALISE_HEIGHT)
#     #     print("corrected positions", proj_positions)
#     record_match(15)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, DIM[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, DIM[1])
    seconds = 0
    while True:
        print(seconds)
        start = time.time()
        ret, frame = cap.read()
        opp_ids, positions = return_opponent_positions(
                                 frame, ortho_proj, opponent_ids)
        print(positions)
        if len(positions)> 0:
            print('Corrected', proj_pos(positions[0], 420))
#         if seconds%2 == 0:
#             girouette_color = girouette(frame, "yellow")
#             print('giro', girouette_color)
        
        seconds+=1        
#         time.sleep(abs(1.0 + start - time.time()))