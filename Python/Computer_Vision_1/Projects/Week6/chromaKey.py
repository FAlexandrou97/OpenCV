# https://medium.com/fnplus/blue-or-green-screen-effect-with-open-cv-chroma-keying-94d4a6ab2743
# Todo: Convert to HSV to make use of the slider
import cv2
import numpy as np

# Read video, get first frame
cap = cv2.VideoCapture('greenscreen-asteroid.mp4')
ret, frame = cap.read()

# Test
imgT = cv2.imread('blemish.png')

# Select region
r = cv2.selectROI("Select Color to Replace", frame)
# Crop drawn region
imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# Get center pixel color of the drawn rectangle
dimensions = imCrop.shape
color = imCrop[int(dimensions[0]/2), int(dimensions[1]/2)]

# Turn all pixels of selected color to black
mask = cv2.inRange(frame, color, color)
masked_asteroid = np.copy(frame)
masked_asteroid[mask != 0] = [0, 0, 0]

mask_on_second_image = np.copy(imgT)
mask_on_second_image = cv2.resize(mask_on_second_image, (1280, 720))
mask_on_second_image[mask == 0] = [0, 0, 0]

final = masked_asteroid + mask_on_second_image
cv2.imshow("new", final)
cv2.waitKey(0)
cv2.destroyAllWindows()


def do_nothing(eventStuff):
    pass

print(type(color))

cv2.namedWindow('Frame')
cv2.createTrackbar("Tolerance Slider", "Frame", 0, 100, do_nothing)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        tolerance = cv2.getTrackbarPos("Tolerance Slider", "Frame")
        # Turn all pixels of selected color to black
        colorMin = color.copy()
        colorMax = color.copy()
        colorMin[0] = colorMin[0] - tolerance
        colorMax[0] = colorMax[0] + tolerance

        mask = cv2.inRange(frame, colorMin, colorMax)
        masked_asteroid = np.copy(frame)
        masked_asteroid[mask != 0] = [0, 0, 0]

        mask_on_second_image = np.copy(imgT)
        mask_on_second_image = cv2.resize(mask_on_second_image, (1280, 720))
        mask_on_second_image[mask == 0] = [0, 0, 0]

        final = masked_asteroid + mask_on_second_image
        cv2.imshow('Frame', final)
        cv2.waitKey(10)
    else:
        break


cap.release()
cv2.destroyAllWindows()