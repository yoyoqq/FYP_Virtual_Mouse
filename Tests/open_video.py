

import cv2
import mediapipe as mp

video = "..\Dataset\Own_dataset\Pointer.mp4"

cap = cv2.VideoCapture(video)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Finishing processing.")
        break
    
    # Convert the frame color to RGB since MediaPipe requires it
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    
    # Process the frame for hand detection
    results = hands.process(frame_rgb)
    
    # Draw the hand annotations on the frame
    frame_rgb.flags.writeable = True
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the processed frame
    cv2.imshow('Processed Video', frame)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()