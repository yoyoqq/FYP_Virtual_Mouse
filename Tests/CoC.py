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
number_of_circles = 3

# PREVIOUS CONFIG FOR THE CoC
# circle length
# inner = 0.15
# middle = 0.3
# exterior = 0.45
# mouse speed 
# inner_mouse_speed = 5
# middle_mouse_speed = 30
# exterior_mouse_speed = 50
# outer_mouse_speed = 100

# width_mid = 960/2
# height_mid = 540/2


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

    # Pointer  
    pointer = Pointer(number_of_circles)
    # inner_circle = pointer.calculate_circle_dimension(inner)
    # middle_circle = pointer.calculate_circle_dimension(middle)
    # exterior_circle = pointer.calculate_circle_dimension(exterior)

    # Delay and coordinate history
    history_length = 16
    gesture_history = deque(maxlen=history_length)

    while True:
        fps = cvFpsCalc.get()
        
        # Process Key (ESC: end) #################################################
        mode = 0
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)
        
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
                    # Convert normalized position to pixel
                    index_finger_tip_x = int(index_finger_tip.x * width)
                    index_finger_tip_y = int(index_finger_tip.y * height)
                    angle = pointer.calculate_angle_with_respect_to_midpoint(index_finger_tip_x, index_finger_tip_y)
                    pointer.move_pointer(index_finger_tip_x, index_finger_tip_y, angle)
                    if mode == 3:
                        # update new center of the CoC depending on the index coordinates
                        pointer.center = [index_finger_tip_x, index_finger_tip_y]
                    # draw 
                    debug_image = pointer.draw_triangle_and_show_angle(debug_image, (index_finger_tip_x, index_finger_tip_y))
                    debug_image = draw_circle(debug_image, pointer.circle_radius, pointer.center)

        # draw info
        debug_image = draw_info(debug_image, fps, mode)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        center_window('Hand Gesture Recognition')

    cap.release()
    cv.destroyAllWindows()



def draw_info(image, fps, mode):
    # FPS
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    # mid point
    # cv.circle(image, (width//2, height//2), radius=5, color=(0, 255, 0), thickness=-1)
    # mode 
    cv.putText(image, "MODE: " + str(mode), (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)

    return image
    
def draw_circle(image, circles_radius: list, center):
    for radius in circles_radius:
        cv.circle(image, center, radius, (0, 255, 0), 2)
    return image

class Pointer():
    def __init__(self, number_of_circles) -> None:
        self.center = [width//2, height//2]
        self.circle_radius = self.create_circles(number_of_circles)
        self.mouse_speed = self.__create_mouse_speed(number_of_circles)
    
    # def moving_area(self):
    #     in_area = False
    #     if (self.center[0] - self.circle_radius[-1] < 0 or
    #             self.center[0] + self.circle_radius[-1] > width or
    #             self.center[1] - self.circle_radius[-1] < 0 or
    #             self.center[1] + self.circle_radius[-1] > height):
    #         in_area = True
    #     return in_area
    
    # def draw_pointer_area(self, image):
    #     # Get total screen size
    #     total_screen_width, total_screen_height = pyautogui.size()
        
    #     # Calculate rectangle coordinates
    #     rect_x = max(0, self.center[0] - self.circle_radius[-1])
    #     rect_y = max(0, self.center[1] - self.circle_radius[-1])
        
    #     # Calculate rectangle dimensions
    #     rect_width = min(total_screen_width - rect_x, 2 * self.circle_radius[-1])
    #     rect_height = min(total_screen_height - rect_y, 2 * self.circle_radius[-1])
        
    #     # Draw rectangle
    #     cv.rectangle(image, (int(rect_x), int(rect_y)), (int(rect_x + rect_width), int(rect_y + rect_height)), (255, 0, 0), 2)
    #     return image

    def move_pointer(self, index_finger_tip_x, index_finger_tip_y, angle):
        center_x, center_y = self.center
        # Calculate distance from the center
        distance = math.sqrt((center_x - index_finger_tip_x) ** 2 + (center_y - index_finger_tip_y) ** 2)
        # Determine speed based on the zone
        speed = 0
        for i in range(len(self.circle_radius)):
            if distance <= self.circle_radius[i]:
                speed = self.mouse_speed[i]
                break
        if distance > self.circle_radius[len(self.circle_radius)-1]:
            speed = self.mouse_speed[len(self.mouse_speed)-1]
    
        # Convert angle to radians and calculate movement deltas
        angle_radians = math.radians(angle)
        delta_x = speed * math.cos(angle_radians)
        delta_y = -speed * math.sin(angle_radians)  # Screen coordinates: y increases downwards
        # Move the mouse pointer by the calculated deltas
        pyautogui.move(delta_x, delta_y)
    
    def draw_triangle_and_show_angle(self, image, fingertip):
        color=(255, 0, 0)
        # Calculate the angle
        angle_degrees_normalized = self.calculate_angle_with_respect_to_midpoint(fingertip[0], fingertip[1])
        # Choose a third point for the triangle (e.g., directly below the midpoint for simplicity)
        third_point = (self.center[0], self.center[1] + 100)
        # Draw lines to form the triangle
        cv.line(image, self.center, fingertip, color, 2)
        cv.line(image, fingertip, third_point, color, 2)
        cv.line(image, third_point, self.center, color, 2)
        # Display the angle
        text_position = (self.center[0] - 10, self.center[1] - 10)  # Slightly above and to the left of the midpoint
        cv.putText(image, "Angle: %s" %(angle_degrees_normalized), text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image
    
    
    def calculate_angle_with_respect_to_midpoint(self, x, y):
        dx = x - self.center[0]
        dy = y - self.center[1]
        # Calculate the angle in radians
        angle_radians = math.atan2(dy, dx)
        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)
        angle_degrees = 360 - angle_degrees
        # Normalize the angle to be between 0 and 360
        angle_degrees_normalized = angle_degrees % 360
        return int(angle_degrees_normalized)
           
    def create_circles(self, number_of_circles):
        circles_diameter = [self.calculate_circle_dimension(0.05)]
        prev = 0.05
        for i in range(number_of_circles):
            circles_diameter.append(self.calculate_circle_dimension(prev + 0.1))
            prev += 0.13
        return circles_diameter
        
    def calculate_circle_dimension(self, fraction):
        # radius = int(height * (1-fraction))
        radius = int(height * fraction)
        return radius

    def __create_mouse_speed(self, number_of_circles):
        speed = [3]
        for i in range(number_of_circles+1):
            speed.append(speed[-1] + 10)
        speed[-1] = 100
        return speed 
    
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 112:  # p
        mode = 3
    return number, mode

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
    
    # PREVIOUS CONFIG FOR THE CoC
    # circle length
    # inner = 0.15
    # middle = 0.3
    # exterior = 0.45
    # mouse speed 
    # inner_mouse_speed = 5
    # middle_mouse_speed = 30
    # exterior_mouse_speed = 50
    # outer_mouse_speed = 100
    
    # pointer = Pointer(7)
    # print(pointer.circle_radius)
    # print(pointer.mouse_speed)
    
    # print(pointer.calculate_circle_dimension(0.10))
    # print(pointer.calculate_circle_dimension(0.20))
    # print(pointer.calculate_circle_dimension(0.30))
    # print(pointer.calculate_circle_dimension(0.40))
    
    # print(pointer.create_circles(7))