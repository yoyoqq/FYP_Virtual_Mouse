#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from typing import *

import mediapipe as mp
import cv2 as cv
import numpy as np
import mediapipe as mp
from cvfpscalc import *

def main():
    # camera 16/9 ratio
    width = 960
    height = 540 
    # rectangle percentages 
    inner = 0.6
    middle = 0.3
    
    
    # calculate rectangles area
    inner_rectangle = calculate_rectangle_dimension(width, height, inner)
    middle_rectangle = calculate_rectangle_dimension(width, height, middle)
    
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
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # draw info
        debug_image = draw_info(debug_image, fps)
        debug_image = draw_rectangles(debug_image, inner_rectangle)
        debug_image = draw_rectangles(debug_image, middle_rectangle)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    return image

def calculate_rectangle_dimension(width, height, new_size) -> List[int]:
    # New dimensions after decreasing size by x%
    new_width = width * (1-new_size)
    new_height = height * (1-new_size)
    # Calculate top-left corner coordinates for the rectangle
    x = (width - new_width) / 2
    y = (height - new_height) / 2
    return [int(x), int(y), int(x+new_width), int(y+new_height)] 

def draw_rectangles(image, coordinates):
    cv.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    main()