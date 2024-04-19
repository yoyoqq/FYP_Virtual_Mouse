import copy
from collections import Counter
from typing import *
import pyautogui

import cv2 as cv
import numpy as np
import mediapipe as mp
from cvfpscalc import *


pyautogui.FAILSAFE = False

#CONFIGURATION

# width = 1920
# height = 1080

# camera 16/9 ratio
width = 960
height = 540

# 4:3
# width = 640
# height = 480

# width = 640
# height = 360

# width = 320
# height = 180

video = "..\Dataset\Own_dataset\Pointer.mp4"

def main():
    # CV2
    # cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap = cv.VideoCapture(video)
    # cap.set(cv.CAP_PROP_FPS, 30)    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    # cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))

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

if __name__ == '__main__':
    main()