import cv2
import time
import math

# Initialize the VideoCapture object
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # Record the start time
    start = time.time()
    # Capture frame-by-frame
    ret, frame = cam.read()
    # Record the end time
    end = time.time()

    # Calculate the FPS
    calc = 1 if end - start == 0 else end - start
    fps = math.ceil(1 / calc)

    # Display the FPS on the frame
    cv2.putText(frame, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cam.release()
cv2.destroyAllWindows()
