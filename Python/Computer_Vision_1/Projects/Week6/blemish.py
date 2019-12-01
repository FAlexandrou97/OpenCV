import cv2
import numpy as np


def click(event, x, y, flags, param):
    global img
    filterRadius = 25
    roi = img[y:y + filterRadius, x:x + filterRadius]
    clickLocation = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        imgBlur = cv2.blur(roi, (3,3), 5)
        sobelxy = cv2.Sobel(imgBlur, cv2.CV_32F, 1, 1)
        mask = 255 * np.ones(roi.shape, roi.dtype)
        img = cv2.seamlessClone(sobelxy, img, mask, clickLocation, cv2.NORMAL_CLONE)
        cv2.imshow("blemish", img)

        # Used for testing in the beginning
        '''
        img = cv2.circle(img, clickLocation, 2, (0,255,0), 2)
        imgBlur = cv2.GaussianBlur(roi,(5,5),7)
        '''


img = cv2.imread("blemish.png")

cv2.imshow("blemish", img)

cv2.setMouseCallback("blemish", click)
cv2.waitKey(0)
