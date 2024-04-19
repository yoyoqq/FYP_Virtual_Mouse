#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from typing import *
import pyautogui
import math

import cv2 as cv
import numpy as np
import mediapipe as mp
from cvfpscalc import *


pyautogui.FAILSAFE = False

#CONFIGURATION
# camera 16/9 ratio
width = 960
height = 540
# Circle percentages 
# circle_size = [0.17, 0.3, 0.5]
inner = 0.15
middle = 0.3
exterior = 0.45

# mouse speed 
# mouse_speed = [5, 30, 70]
inner_mouse_speed = 5
middle_mouse_speed = 30
exterior_mouse_speed = 50
outer_mouse_speed = 100


# PROGRAM VARIABLES
# position of the mouse 
center = [width//2, height//2] 


def main():
    
    # calculate rectangles area
    # inner_rectangle = calculate_rectangle_dimension(width, height, inner)
    # middle_rectangle = calculate_rectangle_dimension(width, height, middle)
    inner_circle = calculate_circle_dimension(inner)
    middle_circle = calculate_circle_dimension(middle)
    exterior_circle = calculate_circle_dimension(exterior)
    
    # Hand model load 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode="use_static_image_mode",
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # CV2
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        fps = cvFpsCalc.get()
        
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        
        # Detection implementation #############################################################
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        results = hands.process(image_rgb)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
                # Get index finger tip coordinates (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                if index_finger_tip:
                    # print(index_finger_tip)
                    # Convert normalized position to pixel
                    index_finger_tip_x = int(index_finger_tip.x * width)
                    index_finger_tip_y = int(index_finger_tip.y * height)
                    # Optionally, use the coordinates for something (e.g., draw a circle at the index finger tip)
                    cv.circle(debug_image, (index_finger_tip_x, index_finger_tip_y), 10, (255, 0, 0), -1)
                    angle = calculate_angle_with_respect_to_midpoint(index_finger_tip_x, index_finger_tip_y)
                    # print(angle)
                    debug_image = draw_triangle_and_show_angle(debug_image, (index_finger_tip_x, index_finger_tip_y))
                    move_pointer(index_finger_tip_x, index_finger_tip_y, angle, inner_circle, middle_circle, exterior_circle)

        # draw info
        debug_image = draw_info(debug_image, fps)
        debug_image = draw_circle(debug_image, inner_circle)
        debug_image = draw_circle(debug_image, middle_circle)
        debug_image = draw_circle(debug_image, exterior_circle)
        # debug_image = draw_rectangles(debug_image, inner_rectangle)
        # debug_image = draw_rectangles(debug_image, middle_rectangle)


        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        center_window('Hand Gesture Recognition')

    cap.release()
    cv.destroyAllWindows()


def draw_info(image, fps):
    # FPS
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    # mid point
    cv.circle(image, (width//2, height//2), radius=5, color=(0, 255, 0), thickness=-1)

    return image

def calculate_rectangle_dimension(width, height, new_size) -> List[int]:
    # New dimensions after decreasing size by x%
    new_width = width * (1-new_size)
    new_height = height * (1-new_size)
    # Calculate top-left corner coordinates for the rectangle
    x = (width - new_width) / 2
    y = (height - new_height) / 2
    return [int(x), int(y), int(x+new_width), int(y+new_height)] 

def calculate_circle_dimension(fraction):
    # radius = int(height * (1-fraction))
    radius = int(height * fraction)
    return radius

def draw_circle(image, radius):
    cv.circle(image, center, radius, (0, 255, 0), 2)
    return image

def draw_rectangles(image, coordinates):
    cv.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 255, 0), 2)
    return image

def calculate_angle_with_respect_to_midpoint(x, y):
    dx = x - center[0]
    dy = y - center[1]
    # Calculate the angle in radians
    angle_radians = math.atan2(dy, dx)
    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)
    angle_degrees = 360 - angle_degrees
    # Normalize the angle to be between 0 and 360
    angle_degrees_normalized = angle_degrees % 360
    return int(angle_degrees_normalized)


def draw_triangle_and_show_angle(image, fingertip):
    color=(255, 0, 0)
    # Calculate the angle
    angle_degrees_normalized = calculate_angle_with_respect_to_midpoint(fingertip[0], fingertip[1])
    # Choose a third point for the triangle (e.g., directly below the midpoint for simplicity)
    third_point = (center[0], center[1] + 100)
    # Draw lines to form the triangle
    cv.line(image, center, fingertip, color, 2)
    cv.line(image, fingertip, third_point, color, 2)
    cv.line(image, third_point, center, color, 2)
    # Display the angle
    text_position = (center[0] - 10, center[1] - 10)  # Slightly above and to the left of the midpoint
    cv.putText(image, "Angle: %s" %(angle_degrees_normalized), text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def move_pointer_to_center():
    # Get the size of the primary monitor
    screen_width, screen_height = pyautogui.size()
    # Calculate the center of the screen
    x, y = screen_width / 2, screen_height / 2
    # Move the mouse pointer to the center of the screen
    pyautogui.moveTo(x, y)
    return [x, y]

# Function to move the OpenCV window to the center of the screen
def center_window(window_name):
    # Get the screen size
    screen_width, screen_height = pyautogui.size()
    # Calculate the center position
    position_x = (screen_width - width) // 2
    position_y = (screen_height - height) // 2
    # Move the OpenCV window to the center
    cv.moveWindow(window_name, position_x, position_y)
    
        
def move_pointer(index_finger_tip_x, index_finger_tip_y, angle, inner_circle_radius, middle_circle_radius, exterior_circle_radius):
    center_x, center_y = center  # Unpack center coordinates
    # Calculate distance from the center
    distance = math.sqrt((center_x - index_finger_tip_x) ** 2 + (center_y - index_finger_tip_y) ** 2)
    # print(distance, inner_circle_radius, middle_circle_radius)
    # Determine speed based on the zone
    if distance <= inner_circle_radius:
        speed = inner_mouse_speed
    elif distance <= middle_circle_radius:
        speed = middle_mouse_speed
    elif distance <= exterior_circle_radius:
        speed = exterior_mouse_speed
    else:
        speed = outer_mouse_speed

    # Convert angle to radians and calculate movement deltas
    angle_radians = math.radians(angle)
    delta_x = speed * math.cos(angle_radians)
    delta_y = -speed * math.sin(angle_radians)  # Screen coordinates: y increases downwards

    # Move the mouse pointer by the calculated deltas
    pyautogui.move(delta_x, delta_y)

    
    
if __name__ == '__main__':
    x, y = move_pointer_to_center()
    main()