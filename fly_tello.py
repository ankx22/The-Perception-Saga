import os
import time
import sys

import cv2
import numpy as np

from djitellopy import Tello
from ultralytics import YOLO

#----------pnp visualizations-----------------------------------

def draw_axis(img, rotvec, tvec, K):
    # unit is mm
    # rotvec, _ = cv2.Rodrigues(R)
    points = np.float32([[20, 0, 0], [0, 20, 0], [0, 0, 20], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotvec, tvec, K, (0, 0, 0, 0))
    axisPoints = axisPoints.astype(int)
    # print("printing axis points",)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3) # Blue is x axis
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3) # Green is Y axis
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3) # Red is z axis
    return img


#------Corner infer funcs--------------------------------------
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def get_corners(img, mask):
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((3, 3), np.uint8)
    
    # The first parameter is the original image, 
    # kernel is the matrix with which image is convolved and third parameter is the number 
    # iterations will determine how much you want to erode/dilate a given image. 
    img_erosion = cv2.erode(mask, kernel, iterations=15) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=15) 

    img_resized = resize_with_aspect_ratio(img, width=900)
    resized_image = resize_with_aspect_ratio(mask, width=900) # You can adjust the width as needed
    resized_image_erosion = resize_with_aspect_ratio(img_erosion, width=900) # You can adjust the width as needed
    resized_image_dilation = resize_with_aspect_ratio(img_dilation, width=900) # You can adjust the width as needed

    contours, _ = cv2.findContours(np.uint8(resized_image_dilation), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    resized_image_dilation_color = cv2.cvtColor(resized_image_dilation, cv2.COLOR_GRAY2BGR)

    # print(contours)
    corners = []
    for contour in contours:
        # Approximate polygon and ensure it has 4 corners
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # Draw circles on the corner points
            for point in approx:
                x,y = point[0]
                cv2.circle(img_resized, (int(x), int(y)), 7, (0, 0, 255), -1)
                corners.append((x,y))
    
    # cv2.imshow('Input', resized_image) 
    # cv2.imshow('Erosion', resized_image_erosion) 
    # cv2.imshow('Dilation', resized_image_dilation_color) 
    # cv2.imshow('Detected contours', img) 

    corner_filename = current_path + str("/corners/") + f"frame{i}.png"
    cv2.imwrite(corner_filename, img_resized)

    # cv2.waitKey(0)
    return corners,img_resized

#--------------------------------------------------------------------------------------------------------
# func to order the corners:
# def order_corners(corners):
#     first_point = corners[0]
#     second_point = corners[1]
#     third_point = corners[2]
#     fourth_point = corners[3]
def order_points(points):
    # Sort points by x-coordinate (leftmost will be top-left, rightmost will be top-right)
    sorted_points = sorted(points, key=lambda x: x[0])
    
    # left most and right most points
    left1 = sorted_points[0]
    left2 = sorted_points[1]

    right1 = sorted_points[-1]
    right2 = sorted_points[-2]

    if left1[1] > left2[1]:
        top_left = left1
        bottom_left = left2
    else:
        top_left = left2

    if left1[1] > left2[1]:
        top_left = left1
    else:
        top_left = left2
    # Identify top-left and top-right points
    top_left = sorted_points[0]
    top_right = sorted_points[1]
    
    # Calculate the distance between the top-left and top-right points
    # The point with smaller y-coordinate is top-left, the other is top-right
    if top_left[1] > top_right[1]:
        top_left, top_right = top_right, top_left
    
    # Identify bottom-left and bottom-right points
    bottom_left = sorted_points[2] if sorted_points[2][1] > sorted_points[3][1] else sorted_points[3]
    bottom_right = sorted_points[3] if sorted_points[2][1] > sorted_points[3][1] else sorted_points[2]
    
    return (top_left, bottom_left, bottom_right, top_right)

def get_axis(img,corners):
    K = np.array([[1381.2153584922428,0.0,655.1643687241123],[0.0,1388.0303218031256,360.7844112495259],[0.0,0.0,1.0]])
    # points_2D = np.array([(128,233),(208,208),(239,320),(158,340)], dtype="double")
    points_2D = np.array([corners], dtype="double")
    points_3D = np.array([
                        (-50.8 ,45.72,0),     # First
                        (-50.8 ,-45.72,0),  # Second
                        (50.8 ,-45.72,0),# Third
                        (50.8 ,45.72,0) # Fourth 
                        ])
    dist_coeffs = np.zeros((4,1))
    success, rotation_vector, translation_vector = cv2.solvePnP(points_3D, points_2D, K, dist_coeffs, flags=0)
    image = draw_axis(img,rotation_vector,translation_vector,K)
    axis_filename = current_path + str("axes/") + f"frame{i}.png"
    cv2.imwrite(axis_filename, image)
    return rotation_vector, translation_vector

try:
    current_path = os.path.abspath(__file__)
    current_path = current_path.replace("fly_tello.py","")
    model_path = current_path + str("weights/last.pt")
    model = YOLO(model_path)

    # # To run the model
    # results = model(img)
    # mask = results[0].masks.data
    # mask = mask.cpu().numpy()*255
    # mask = cv2.resize(mask[0], (W, H))
    # cv2.imwrite('./output.png', mask)

    # # To get corners
    # corner_img = get_corners(img,mask)


    # Tello drone code template:
    # Check the connection to the drone
    drone = Tello()
    drone.connect()

    # CHECK SENSOR READINGS------------------------------------------------------------------
    print('Altitude ', drone.get_distance_tof())
    print('Battery, ', drone.get_battery())

    # drone.takeoff()
    # time.sleep(2)
    # drone.move_forward(30)
    # time.sleep(2)
    # drone.move_right(30)
    # time.sleep(2)
    # drone.move_back(30)
    # drone.move_back(30)
    # drone.move_back(30)
    # drone.move_back(30)
    # time.sleep(2)
    # drone.move_left(30)
    # time.sleep(2)
    # drone.land()

    drone.streamon()
    # time.sleep(2)
    # drone.takeoff()
    i = 0
    while True:
        frame = drone.get_frame_read().frame
        frame = cv2.resize(frame, (360, 240)) # Windows
        # frame = cv2.resize(frame, (360, 240), interpolation=cv2.INTER_LINEAR) # Ubuntu
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        H, W, _ = frame.shape
        print("shape",(H,W))
        # cv2.imread(frame)
        filename = current_path + str("/frames/") + f"frame{i}.png"
        cv2.imwrite(filename, frame)
        # Running the model on the saved frame
        # # To run the model
        frame = cv2.imread(filename)
        results = model(frame)
        try:
            mask = results[0].masks.data
            mask = mask.cpu().numpy()*255
            print("mask shape",mask.shape)
            mask = cv2.resize(mask[0], (W, H))
            mask_filename = current_path + str("/masks/") + f"frame{i}.png"
            cv2.imwrite(mask_filename, mask)

            # To get corners
            corners, corner_img = get_corners(frame,mask)
            rvec, tvec = get_axis(corner_img,corners)
            # Adding axis to drone and saving image

            i += 1
        except:
            continue

        

        # cv2.imshow('video', frame)
        # key = cv2.waitKey(2) & 0xFF
        # if key == ord('q'):
        #     break
        # elif key == ord('w'):
        #     drone.move_forward(30)
        # elif key == ord('s'):
        #     drone.move_back(30)
        # elif key == ord('a'):
        #     drone.move_left(30)
        # elif key == ord('d'):
        #     drone.move_right(30)
        # elif key == ord('p'):
        #     print(f"Taking picture....\n")
        #     filename = f"frame{i}.png"
        #     cv2.imwrite(filename, frame)
        
    # print(f"Landing....")
    # drone.land()

except KeyboardInterrupt:
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('keyboard interrupt')
    drone.emergency()
    drone.emergency()
    drone.end()
