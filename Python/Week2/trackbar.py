import cv2


# Callback functions
def scaleImage(*args):
    global scaleFactor

    # Perform check if scaleFactor is zero
    if scaleFactor == 0:
        scaleFactor = 1

    # Scale Image according to the scale type
    if scaleType == 1:
        scaleFactor = 1 - args[0] / 100.0
    else:
        scaleFactor = 1 + args[0] / 100.0

    # Resize the image
    scaledImage = cv2.resize(im, None, fx=scaleFactor,
                             fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(windowName, scaledImage)


# Callback functions
def scaleTypeImage(*args):
    global scaleType
    global scaleFactor
    scaleType = args[0]
    scaleFactor = 1 + scaleFactor / 100.0
    if scaleFactor == 0:
        scaleFactor = 1

    scaledImage = cv2.resize(im, None, fx=scaleFactor,
                             fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(windowName, scaledImage)

maxScaleUp = 100
scaleFactor = 1
scaleType = 0
maxType = 1

windowName = "Resize Image"
trackbarValue = "Scale"
trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"

# load an image
im = cv2.imread("truth.png")

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar(trackbarValue, windowName, scaleFactor, maxScaleUp, scaleImage)
cv2.createTrackbar(trackbarType, windowName, scaleType, maxType, scaleTypeImage)

cv2.imshow(windowName, im)

c = cv2.waitKey(0)

cv2.destroyAllWindows()