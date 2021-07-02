import cv2 as cv
import cv2.aruco as aruco
import json
import time
import numpy as np

CENTRAL_POSITION = [1500, 2000]
YELLOW_OFFSET = [-10, 0]
BLUE_OFFSET = [10, 0]
CAMERA_HEIGHT = 1200
BALISE_HEIGHT = 510

COLOR = (255, 0, 0)
aruco_signs_legend = {1: "bleu", 2: "bleu", 3: "bleu", 4: "bleu", 5: "bleu",
                      6: "jaune", 7: 'jaune', 8: 'jaune', 9: 'jaune', 10: 'jaune',
                      42: 'Board center'}
BLUE_TEAM_IDS = list(range(1, 6))
YELLOW_TEAM_IDS = list(range(5, 11))

camera_position = CENTRAL_POSITION + YELLOW_OFFSET

# Load ARUCO parameters
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_4X4_50)
# TODO check where to put this command
aruco_Params = aruco.DetectorParameters_create()
# Recover fisheye camera settings from JSON config file
with open('Fisheye_config.json') as config:
    fisheye_cfg = json.load(config)

mtx = np.asarray(fisheye_cfg['mtx'])
dist = np.asarray(fisheye_cfg['dist'])

DIM = (1280, 720)  # standard dimension of pics from camera
map1, map2 = cv.fisheye.initUndistortRectifyMap(
    mtx, dist, np.eye(3), mtx, DIM, cv.CV_16SC2)

# IN/OUT points used to transform undistort image to orthonormal image (simulate view from the top)
# Values for camera on YELLOW side:
# TODO current values are not giving proper results
# yellow_in_points = np.float32([[269, 77],
#                                [1168, 178],
#                                [1089, 560],
#                                [159, 438]])
# 
# yellow_out_points = np.float32([[400, 800],
#                                 [2500, 800],
#                                 [2100, 1700],
#                                 [600, 1700]])
yellow_in_points = np.float32([[232, 534],
                               [1065, 353],
                               [1100, 43],
                               [306, 4]])

yellow_out_points = np.float32([[700, 1700],
                                [2000, 1400],
                                [2500, 500],
                                [500, 600]])
# Values for camera on BLUE side:
# TODO need to be defined (currently using YELLOW values)
blue_in_points = np.float32([[269, 77],
                             [1168, 178],
                             [1089, 560],
                             [159, 438]])

blue_out_points = np.float32([[400, 800],
                              [2500, 800],
                              [2100, 1700],
                              [600, 1700]])


# Function that takes a fisheye picture, undistort and project it in an orthornaml system (view from the top)

def setup_for_match(team_color):
    if team_color == "yellow":
        ortho_proj = cv.getPerspectiveTransform(
            yellow_in_points, yellow_out_points)
        opponents_ids = BLUE_TEAM_IDS

    elif team_color == "blue":
        ortho_proj = cv.getPerspectiveTransform(
            blue_in_points, blue_out_points)
        opponents_ids = YELLOW_TEAM_IDS

    else:
        print("ERROR IN THE COLOR NAME OF THE TEAM")
    return ortho_proj, opponents_ids


def undistort_and_project_image(fisheye_image):
    processed_img = cv.remap(
        fisheye_image, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    cv.imwrite('undistorted_yellow/verres_undist2_28.png',processed_img)
    processed_img = cv.warpPerspective(
        processed_img, ORTHO_PROJ, (3000, 2000), flags=cv.INTER_LINEAR)
    cv.imwrite('undistorted_yellow/verres_warp2_28.png',processed_img)
    return processed_img


def get_arucos(projected_img):
    corners, ids, rejected = aruco.detectMarkers(
        projected_img, ARUCO_DICT, parameters=aruco_Params)
    img_with_markers = cv.aruco.drawDetectedMarkers(
        projected_img, corners, ids)
    return img_with_markers, corners, ids

# Not used


def get_opponent_positions(id, center):
    opponent_positions = []
    if id in OPPONENT_IDS:
        opponent_positions.add(center)
    return opponent_positions


def get_team_positions():
    pass


def return_opponent_positions(fisheye_image):
    processed_image = undistort_and_project_image(fisheye_image)
    img_with_markers, corners, ids = get_arucos(processed_image)
    opponent_positions = []
    for i, val in enumerate(corners):
        try:
            sign = aruco_signs_legend[int(ids[i])]
            id_min = val[0].sum(axis=1).argmin()
            id_max = val[0].sum(axis=1).argmax()
            UL_x, UL_y = int(val[0][id_min][0]), int(val[0][id_min][1])
            DR_x, DR_y = int(val[0][id_max][0]), int(val[0][id_max][1])
            center_x, center_y = (DR_x+UL_x)//2, (DR_y+UL_y)//2
            cv.putText(img_with_markers, sign, (UL_x, UL_y-5),
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, COLOR, 6)
            if ids[i] in OPPONENT_IDS:
                opponent_positions.append([ids[i][0], [center_x, center_y]])

        except KeyError:
            continue
    cv.imshow('treated Image', scale_pic(img_with_markers))
    cv.imwrite('out/warp_yellow.jpg', img_with_markers)
    cv.waitKey(0)
    return opponent_positions

# this function may not be needed in production except if we want to return the image for live streaming


def scale_pic(src, ratio=50):
    scale_percent = ratio
    # calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # resize image
    output = cv.resize(src, dsize)

    return output

# function to change coordinate system from board ([0,0] in left upper corner) to camera centered coordinate system


def coordinates_board2camera(coordinates):
    return np.array([coordinates[0] - camera_position[0], camera_position[1] - coordinates[1]])

# function to change camera centered coordinate system to coordinate system from board ([0,0] in left upper corner)


def coordinates_camera2board(coordinates):
    return np.array([coordinates[0] + camera_position[0], camera_position[1] - coordinates[1]])

# Function to transform cartesian coordinates in polar coordinates


def cart2pol(position):
    rho = np.linalg.norm(position)
    phi = np.arctan2(position[1], position[0])
    return np.array([rho, phi])

# Function to transform polar coordinates in cartesian coordinates


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])

# Function to project coordinates from elevated targets (specifically Aruco signs on balise)
# @param: object_pos is a 1D np.array with x,y coordinates of the object in the board coordinate system ([0,0] in left upper corner)
# @param: object_h is the height of the object above the board


def proj_pos(object_pos, object_h):
    coord_camera_space = coordinates_board2camera(object_pos)
    r, theta = cart2pol(coord_camera_space)
    proj_r = r * object_h / CAMERA_HEIGHT
    proj_vec = pol2cart(proj_r, theta)
    proj_position = coordinates_camera2board(proj_vec)
    return proj_position


if __name__ == "__main__":
    ###### MATCH SETUP ####
    ORTHO_PROJ, OPPONENT_IDS = setup_for_match("yellow")

    # GENERAL SETUP, TO BE KEPT FOR EACH OPTIONS:    ##############""

    # SINGLE IMAGE: Load image to check for single image mode, just uncomment the next 2 paragraphs to use on single image
    img = cv.imread('Robot_signs/Full_board1.PNG')

    img6 = cv.imread("coordinates/Blue_2000_1300.png")
    img7 = cv.imread("updated_pos/board_img_1.png")
    img6 = cv.imread("coordinates/balise gauche jaune/verres_config2_28.png")

    # for i in range(1,6):
    #     curr_img = cv.imread(f"angle_selection/trait_{i}.png")
    # #     img2 = cv.imread(f"test_pics/prev/board_img_{i}.png")
    #     curr_img = cv.remap(curr_img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # #     img2 = cv.remap(img2, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    #     cv.imwrite(f'angle_selection/undistort_trait_{i}.jpg', curr_img)
    #     cv.imwrite(f'train/result_1{i}.jpg', img2)
    #
    curr_img = img6
    # cv.imshow('Result', curr_img)
    positions = return_opponent_positions(curr_img)
    print(positions)
    proj_positions = proj_pos(positions[0][1], BALISE_HEIGHT)
    print(proj_positions)

    # Code to check on a single image
    # curr_img = cv.remap(curr_img, map1, map2,
    #                     interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # corners, ids, rejected = aruco.detectMarkers(
    #     curr_img, ARUCO_DICT, parameters=aruco_Params)
    # curr_img = cv.aruco.drawDetectedMarkers(curr_img, corners, ids)

    # # init_Lsupport = np.float32([[286, 77],
    # #             [896, 79],
    # #             [1111, 445],
    # #             [272, 511]])
    # #
    # # out_Lsupport = np.float32([[400, 900],
    # #             [1900, 900],
    # #             [2100, 1700],
    # #             [800, 1800]])

    # # ortho_proj = cv.getPerspectiveTransform(corners_42, coord_42)
    # ortho_proj = cv.getPerspectiveTransform(init_points, out_points)
    # out = cv.warpPerspective(curr_img, ortho_proj,
    #                          (3000, 2000), flags=cv.INTER_LINEAR)
    # cv.imwrite('out/Warp_Lsupport.jpeg', out)
    # cv.imwrite('out/Undistor_Lsupport.jpeg', curr_img)
    # out = scale_pic(out, 40)
    # for i, val in enumerate(corners):
    #     try:
    #         sign = aruco_signs_legend[int(ids[i])]
    #         id_min = val[0].sum(axis=1).argmin()
    #         id_max = val[0].sum(axis=1).argmax()
    #         UL_x, UL_y = int(val[0][id_min][0]), int(val[0][id_min][1])
    #         DR_x, DR_y = int(val[0][id_max][0]), int(val[0][id_max][1])
    #         center_x, center_y = (DR_x+UL_x)//2, (DR_y+UL_y)//2
    #         cv.rectangle(curr_img, (UL_x, UL_y), (DR_x, DR_y), COLOR, 3)
    #         cv.putText(curr_img, sign, (UL_x, UL_y-5),
    #                    cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR, 2)
    # #         cv.circle(curr_img,(center_x,center_y), 2, (0,255,255), -1)
    #     except KeyError:
    #         continue
    # curr_img = scale_pic(curr_img, 50)
    # cv.imshow('Result', curr_img)
    # cv.imshow('warp', out)
    # cv.waitKey(0)

    # FROM WEBCAM STREAM

    # cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH,DIM[0])
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT,DIM[1])
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter('POC_Signs_reco.avi',fourcc, 20.0, (640,480))
    #
    # while True:
    #
    #     # GET CAMERA IMAGE
    #     success, img = cap.read()
    #     # DISPLAY THE DETECTED OBJECTS
    #     corners, ids , rejected= aruco.detectMarkers(img,ARUCO_DICT,parameters=aruco_Params)
    #     aruco.drawDetectedMarkers(img,corners)
    #     for i,val in enumerate(corners):
    #         try:
    #             curr_id=int(ids[i])
    #             if curr_id in aruco_signs_legend:
    #                 sign=aruco_signs_legend[int(ids[i])]
    #             else:
    #                 sign=str(curr_id)
    #             id_min=val[0].sum(axis=1).argmin()
    #             id_max=val[0].sum(axis=1).argmax()
    #             UL_x,UL_y=int(val[0][id_min][0]),int(val[0][id_min][1])
    #             DR_x,DR_y=int(val[0][id_max][0]),int(val[0][id_max][1])
    #             center_x,center_y=(DR_x+UL_x)//2,(DR_y+UL_y)//2
    # #             cv.rectangle(img,(UL_x,UL_y),(DR_x,DR_y),COLOR,2)
    #             cv.putText(img,sign,(UL_x,UL_y-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,COLOR,2)
    # #             cv.circle(img,(center_x,center_y), 2, (0,255,255), -1)
    #         except KeyError:
    #             continue
    #
    #     cv.imshow("Result", img)
    #     dst=cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    # #     out.write(img)
    #     cv.imshow("Straightened image", dst)
    #
    # #     if cv.waitKey(1) and 0xFF == ord("p"):
    # #         print("A")
    # #         cv.imwrite("angle_selection/original_img", img)
    # #         cv.imwrite("angle_selection/undistorted_img", dst)
    #
    #     if cv.waitKey(1) & 0xFF == ord('q'):
    #         print("A")
    #         break

    # from sample video

    # cap = cv.VideoCapture('CDF_sample.mp4')
    # speed=25
    #
    # while (cap.isOpened()):
    #
    #     # GET CAMERA IMAGE
    #     success, img = cap.read()
    #
    #     # DISPLAY THE DETECTED OBJECTS
    #     corners, ids , rejected= aruco.detectMarkers(img,ARUCO_DICT,parameters=aruco_Params)
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
