<<<<<<< HEAD
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import math
import pyautogui

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    args = parser.parse_args()

    return args


def main():
    # POINTER CONFIG 
    number_of_circles = 4
    mouse_increments = 10
    circle_increment = 0.13

    # Argument parsing #################################################################
    args = get_args()

    width = 960
    height = 540

    
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    # cap = cv.VideoCapture(cap_device)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    mp_drawing = mp.solutions.drawing_utils


    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)


    # POINTER 
    pointer = Pointer(width, height, number_of_circles, mouse_increments, circle_increment)
    # SCROLL

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
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
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # pre_processed_point_history_list = pre_process_point_history(
                #     debug_image, point_history)
                # Write to the dataset file
                # logging_csv(number, mode, pre_processed_landmark_list,
                logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                # point_history_len = len(pre_processed_point_history_list)
                # if point_history_len == (history_length * 2):
                #     finger_gesture_id = point_history_classifier(
                #         pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                # print(most_common_fg_id)

                # MAKE GESTURE
                if keypoint_classifier_labels[hand_sign_id] == "Nothing": pass
                elif keypoint_classifier_labels[hand_sign_id] == "Pointer":
                    # print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    # Convert normalized position to pixel
                    index_finger_tip_x = int(index_finger_tip.x * width)
                    index_finger_tip_y = int(index_finger_tip.y * height)
                    # print(index_finger_tip_x, index_finger_tip_y)
                    angle = pointer.calculate_angle_with_respect_to_midpoint(index_finger_tip_x, index_finger_tip_y)
                    if pointer.move_pointer_delay():
                        pointer.move_pointer(index_finger_tip_x, index_finger_tip_y, angle)
                    if mode == 3:
                        # update new center of the CoC depending on the index coordinates
                        pointer.center = [index_finger_tip_x, index_finger_tip_y]
                        pointer.update_center = False
                    # draw 
                    debug_image = pointer.draw_triangle_and_show_angle(debug_image, (index_finger_tip_x, index_finger_tip_y))
                    debug_image = draw_circle(debug_image, pointer.circle_radius, pointer.center)


                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                # debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    # point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        # debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        # center_window('Hand Gesture Recognition', width, height)

    cap.release()
    cv.destroyAllWindows()


class Pointer():
    def __init__(self, width, height, number_of_circles, mouse_increments, circle_increment, pointer_delay = 1) -> None:
        self.width = width
        self.height = height
        self.number_of_circles = number_of_circles
        self.mouse_increments = mouse_increments
        self.circle_increment = circle_increment
        self.center = [width//2, height//2]
        self.circle_radius = self.create_circles(number_of_circles)
        self.mouse_speed = self.__create_mouse_speed(number_of_circles)
        self.pointer_delay = [0, pointer_delay] # current delay, total delay
        self.update_center = True
        self.click_delay = 20
        self.click_cur_delay = 0
    
    def move_pointer_delay(self) -> bool:
        if self.pointer_delay[0] == self.pointer_delay[1]:
            self.pointer_delay[0] = 0
            return True
        self.pointer_delay[0] += 1
        return False

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
        # ? changed for the delay 
        # pyautogui.move(delta_x, delta_y)
        pyautogui.move(delta_x, delta_y, duration=0.1)
    
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
            circles_diameter.append(self.calculate_circle_dimension(prev + self.circle_increment))
            prev += self.circle_increment
        return circles_diameter
        
    def calculate_circle_dimension(self, fraction):
        # radius = int(height * (1-fraction))
        radius = int(self.height * fraction)
        return radius

    def __create_mouse_speed(self, number_of_circles):
        speed = [0]
        for i in range(number_of_circles+1):
            speed.append(speed[-1] + self.mouse_increments)
        speed[-1] = 100
        return speed 
    
    def click(self):
        self.click_cur_delay += 1
        if self.click_cur_delay >= self.click_delay:
            self.click_cur_delay = 0
            return True
        return False
    

def draw_circle(image, circles_radius: list, center):
    for radius in circles_radius:
        cv.circle(image, center, radius, (0, 255, 0), 2)
    return image 

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # save keypoint 
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 112:  # p
        mode = 3
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    # temp_landmark_list = temp_landmark_list[2:]
    # print(temp_landmark_list)
    return temp_landmark_list


# def logging_csv(number, mode, landmark_list, point_history_list):
def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    # if mode == 2 and (0 <= number <= 9):
    #     csv_path = 'model/point_history_classifier/point_history.csv'
    #     with open(csv_path, 'a', newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([number, *point_history_list])
    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # if finger_gesture_text != "":
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image



def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
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
def center_window(window_name, width, height):
    # Get the screen size
    screen_width, screen_height = pyautogui.size()
    # Calculate the center position
    position_x = (screen_width - width) // 2
    position_y = (screen_height - height) // 2
    # Move the OpenCV window to the center
    cv.moveWindow(window_name, position_x, position_y) 

if __name__ == '__main__':
    x, y = move_pointer_to_center()
=======
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import math
import pyautogui

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    args = parser.parse_args()

    return args


def main():
    # POINTER CONFIG 
    number_of_circles = 4
    mouse_increments = 10
    circle_increment = 0.13

    # Argument parsing #################################################################
    args = get_args()

    width = 960
    height = 540

    
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    # cap = cv.VideoCapture(cap_device)
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    mp_drawing = mp.solutions.drawing_utils


    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)


    # POINTER 
    pointer = Pointer(width, height, number_of_circles, mouse_increments, circle_increment)
    # SCROLL

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
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
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # pre_processed_point_history_list = pre_process_point_history(
                #     debug_image, point_history)
                # Write to the dataset file
                # logging_csv(number, mode, pre_processed_landmark_list,
                logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                # point_history_len = len(pre_processed_point_history_list)
                # if point_history_len == (history_length * 2):
                #     finger_gesture_id = point_history_classifier(
                #         pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                # print(most_common_fg_id)

                # MAKE GESTURE
                if keypoint_classifier_labels[hand_sign_id] == "Nothing": pass
                elif keypoint_classifier_labels[hand_sign_id] == "Pointer":
                    # print(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    # Convert normalized position to pixel
                    index_finger_tip_x = int(index_finger_tip.x * width)
                    index_finger_tip_y = int(index_finger_tip.y * height)
                    # print(index_finger_tip_x, index_finger_tip_y)
                    angle = pointer.calculate_angle_with_respect_to_midpoint(index_finger_tip_x, index_finger_tip_y)
                    if pointer.move_pointer_delay():
                        pointer.move_pointer(index_finger_tip_x, index_finger_tip_y, angle)
                    if mode == 3:
                        # update new center of the CoC depending on the index coordinates
                        pointer.center = [index_finger_tip_x, index_finger_tip_y]
                        pointer.update_center = False
                    # draw 
                    debug_image = pointer.draw_triangle_and_show_angle(debug_image, (index_finger_tip_x, index_finger_tip_y))
                    debug_image = draw_circle(debug_image, pointer.circle_radius, pointer.center)


                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                # debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    # point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        # debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)
        # center_window('Hand Gesture Recognition', width, height)

    cap.release()
    cv.destroyAllWindows()


class Pointer():
    def __init__(self, width, height, number_of_circles, mouse_increments, circle_increment, pointer_delay = 1) -> None:
        self.width = width
        self.height = height
        self.number_of_circles = number_of_circles
        self.mouse_increments = mouse_increments
        self.circle_increment = circle_increment
        self.center = [width//2, height//2]
        self.circle_radius = self.create_circles(number_of_circles)
        self.mouse_speed = self.__create_mouse_speed(number_of_circles)
        self.pointer_delay = [0, pointer_delay] # current delay, total delay
        self.update_center = True
        self.click_delay = 20
        self.click_cur_delay = 0
    
    def move_pointer_delay(self) -> bool:
        if self.pointer_delay[0] == self.pointer_delay[1]:
            self.pointer_delay[0] = 0
            return True
        self.pointer_delay[0] += 1
        return False

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
        # ? changed for the delay 
        # pyautogui.move(delta_x, delta_y)
        pyautogui.move(delta_x, delta_y, duration=0.1)
    
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
            circles_diameter.append(self.calculate_circle_dimension(prev + self.circle_increment))
            prev += self.circle_increment
        return circles_diameter
        
    def calculate_circle_dimension(self, fraction):
        # radius = int(height * (1-fraction))
        radius = int(self.height * fraction)
        return radius

    def __create_mouse_speed(self, number_of_circles):
        speed = [0]
        for i in range(number_of_circles+1):
            speed.append(speed[-1] + self.mouse_increments)
        speed[-1] = 100
        return speed 
    
    def click(self):
        self.click_cur_delay += 1
        if self.click_cur_delay >= self.click_delay:
            self.click_cur_delay = 0
            return True
        return False
    

def draw_circle(image, circles_radius: list, center):
    for radius in circles_radius:
        cv.circle(image, center, radius, (0, 255, 0), 2)
    return image 

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # save keypoint 
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 112:  # p
        mode = 3
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    # temp_landmark_list = temp_landmark_list[2:]
    # print(temp_landmark_list)
    return temp_landmark_list


# def logging_csv(number, mode, landmark_list, point_history_list):
def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    # if mode == 2 and (0 <= number <= 9):
    #     csv_path = 'model/point_history_classifier/point_history.csv'
    #     with open(csv_path, 'a', newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([number, *point_history_list])
    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # if finger_gesture_text != "":
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image



def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
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
def center_window(window_name, width, height):
    # Get the screen size
    screen_width, screen_height = pyautogui.size()
    # Calculate the center position
    position_x = (screen_width - width) // 2
    position_y = (screen_height - height) // 2
    # Move the OpenCV window to the center
    cv.moveWindow(window_name, position_x, position_y) 

if __name__ == '__main__':
    x, y = move_pointer_to_center()
>>>>>>> de45f1c6053641d789f9a5f96df63f43c3afb941
    main()