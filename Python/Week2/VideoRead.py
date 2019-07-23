# Import module
import cv2

cap = cv2.VideoCapture('chaplin.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        cv2.waitKey(25)

    # Break the loop
    else:
        break
