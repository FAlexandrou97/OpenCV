import cv2
import matplotlib.pyplot as plt
import numpy as np


def cartoonify(image, arguments=0):
    ### YOUR CODE HERE
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pencilSketchImage = cv2.adaptiveThreshold(imageGray, 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY,
                                              blockSize=9,
                                              C=7)

    cartoonImage = cv2.bitwise_and(image, image, mask=pencilSketchImage)

    return cartoonImage

image = cv2.imread("trump.jpg")

cartoonImage = cartoonify(image)
cv2.imshow("cartoon", cartoonImage)
cv2.waitKey(0)