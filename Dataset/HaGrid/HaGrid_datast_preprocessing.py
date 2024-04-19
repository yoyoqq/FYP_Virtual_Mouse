"""
DATASET PRE PROCESSING FOR THE HAGRID DATASET 

Works in the base folder path     BASE_FOLDER = "./"

1. MOVE THIS FILE TO THE BASE FOLDER
2. INSTALL THE DATASET ->  https://www.kaggle.com/datasets/kapitanov/hagrid/data
3. DELETE THE DATA IN THE KEYPOINT.CSV
4. RUN
"""

import os
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
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

hand_detected = False
# Argument parsing #################################################################
args = get_args()

width = 960
height = 540

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

use_brect = True

# Model load #############################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)
mp_drawing = mp.solutions.drawing_utils


# Read labels ###########################################################
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

def main(idx, file_path):
    hand_detected = False
    # Camera preparation ###############################################################
    cap = cv.VideoCapture(file_path)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    #  ########################################################################
    mode = 0


    number = idx
    mode = 1

    # Camera capture #####################################################
    ret, image = cap.read()
    if not ret:
        return 0
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation #############################################################
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    #  ####################################################################
    if results.multi_hand_landmarks is not None:
        if len(results.multi_hand_landmarks) != 1: return 0
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
            hand_detected = True

    cap.release()
    cv.destroyAllWindows()
    return hand_detected




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


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

    return


if __name__ == '__main__':

    BASE_DIR = r"Dataset\HaGrid" 

    UNDER_DIR = os.listdir(BASE_DIR)
    total_images = 0
    processed_images = 0

    idx = 0
    # Iterate through each category or identifier directory in the base directory
    for folder in UNDER_DIR:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue  # Skip if it's not a directory
        # List files directly under each subdirectory
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                if main(idx, file_path):  
                    processed_images += 1
                # print(idx, file_path)  # Print the index (0) and file path
                total_images += 1
        idx += 1

    print(f"Total images: {total_images}, Processed images: {processed_images}")