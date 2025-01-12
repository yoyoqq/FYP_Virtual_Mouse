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
# rectangle percentages 
inner = 0.17
middle = 0.3
# mouse speed 
inner_mouse_speed = 5
middle_mouse_speed = 30
exterior_mouse_speed = 70

# PROGRAM VARIABLES
# position of the mouse 
center = [width//2, height//2] 


def main():
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

        # draw info
        debug_image = draw_info(debug_image, fps)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_info(image, fps):
    # FPS
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

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
    

    
    
if __name__ == '__main__':
    x, y = move_pointer_to_center()
    main()