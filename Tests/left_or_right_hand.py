import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Dynamic mode
    max_num_hands=2,          # Maximum number of hands to detect
    min_detection_confidence=0.5,  # Detection confidence
    min_tracking_confidence=0.5)   # Tracking confidence

# Initialize OpenCV for capturing video from the webcam.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)

    # Draw the hand annotations on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Determine if the hand is left or right
            hand_label = handedness.classification[0].label
            
            if hand_label == "Right": print("Right")
            if hand_label == "Left": print("Left")
            
            # Set color based on hand type
            color = (255, 0, 0) if hand_label == 'Right' else (0, 0, 255)
            
            # Draw circles on the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 15, color, -1)

    # Show the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
