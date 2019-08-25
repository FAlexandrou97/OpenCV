Assignment
Part - A
Step
1: Read
Image

import cv2

import matplotlib.pyplot as plt

from dataPath import DATA_PATH

import numpy as np

% matplotlib
inline

import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)

matplotlib.rcParams['image.cmap'] = 'gray'

# Image path

imagePath = DATA_PATH + "images/CoinsA.png"

# Read image

# Store it in the variable image

###

### YOUR CODE HERE

###

image = cv2.imread(imagePath)

imageCopy = image.copy()

plt.imshow(image[:, :, ::-1]);

plt.title("Original Image")

Text(0.5, 1, 'Original Image')

Step
2.1: Convert
Image
to
Grayscale

# Convert image to grayscale

# Store it in the variable imageGray

###

### YOUR CODE HERE

###

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 12))

plt.subplot(121)

plt.imshow(image[:, :, ::-1]);

plt.title("Original Image")

plt.subplot(122)

plt.imshow(imageGray);

plt.title("Grayscale Image");

Step
2.2: Split
Image
into
R, G, B
Channels

# Split cell into channels

# Store them in variables imageB, imageG, imageR

###

### YOUR CODE HERE

###

imageB, imageG, imageR = cv2.split(image)

​

plt.figure(figsize=(20, 12))

plt.subplot(141)

plt.imshow(image[:, :, ::-1]);

plt.title("Original Image")

plt.subplot(142)

plt.imshow(imageB);

plt.title("Blue Channel")

plt.subplot(143)

plt.imshow(imageG);

plt.title("Green Channel")

plt.subplot(144)

plt.imshow(imageR);

plt.title("Red Channel");

Step
3.1: Perform
Thresholding

You
will
have
to
carry
out
this
step
with different threshold values to see which one suits you the most.Do not remove those intermediate images and make sure to document your findings.

###

### YOUR CODE HERE

###

thresh = 50

maxValue = 200

th, threshImageGray = cv2.threshold(imageGray, thresh, maxValue, cv2.THRESH_BINARY)

plt.imshow(threshImageGray[:, ::-1]);

plt.title("Thresholded Gray Image")

th, threshImageB = cv2.threshold(imageB, thresh, maxValue, cv2.THRESH_BINARY)

th, threshImageG = cv2.threshold(imageG, thresh, maxValue, cv2.THRESH_BINARY)

th, threshImageR = cv2.threshold(imageR, thresh, maxValue, cv2.THRESH_BINARY)

# Display the thresholded image

###

### YOUR CODE HERE

###

plt.figure(figsize=(20, 12))

plt.subplot(141)

plt.imshow(image[:, :, ::-1]);

plt.title("Thresholded Original Image")

plt.subplot(142)

plt.imshow(threshImageB);

plt.title("Thresholded Blue Channel")

plt.subplot(143)

plt.imshow(threshImageG);

plt.title("Thresholded Green Channel")

plt.subplot(144)

plt.imshow(threshImageR);

plt.title("Thresholded Red Channel");

Step
3.2: Perform
morphological
operations

You
will
have
to
carry
out
this
step
with different kernel size, kernel shape and morphological operations to see which one ( or more) suits you the most.Do not remove those intermediate images and make sure to document your findings.

###

### YOUR CODE HERE

###

​

# Decided to choose the Green Channel thresholded image.

​

​

openingSize = 1

​

# Selecting a elliptical kernel

element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,

                                    (2 * openingSize + 1, 2 * openingSize + 1),

                                    (openingSize, openingSize))

###

### YOUR CODE HERE

###

​

imageOpened = cv2.morphologyEx(threshImageG, cv2.MORPH_OPEN,

                               element, iterations=2)

​

plt.imshow(imageOpened);

plt.title("Opened Image")

​

Text(0.5, 1, 'Opened Image')

# Display all the images

# you have obtained in the intermediate steps

###

### YOUR CODE HERE

###

imageClosed = cv2.morphologyEx(imageOpened, cv2.MORPH_CLOSE,

                               element, iterations=3)

​

plt.imshow(imageClosed);

plt.title("Closed Image")

Text(0.5, 1, 'Closed Image')

# Get structuring element/kernel which will be used for dilation

###

### YOUR CODE HERE

###

​

###

### YOUR CODE HERE

###

finalImage = cv2.morphologyEx(imageClosed, cv2.MORPH_OPEN,

                              element, iterations=28)

​

plt.imshow(finalImage);

plt.title("final Image")

​

​

Text(0.5, 1, 'final Image')

# My result above compared to assignment result below

​

Step
4.1: Create
SimpleBlobDetector

# Set up the SimpleBlobdetector with default parameters.

params = cv2.SimpleBlobDetector_Params()

​

params.blobColor = 255

​

params.minDistBetweenBlobs = 2

​

# Filter by Area.

params.filterByArea = False

​

# Filter by Circularity

params.filterByCircularity = True

params.minCircularity = 0.8

​

# Filter by Convexity

params.filterByConvexity = True

params.minConvexity = 0.8

​

# Filter by Inertia

params.filterByInertia = True

params.minInertiaRatio = 0.8

# Create SimpleBlobDetector

detector = cv2.SimpleBlobDetector_create(params)

Step
4.2: Detect
Coins
Hints

Use
detector.detect(image)
to
detect
the
blobs(coins).The
output
of
the
function is a
list
of
keypoints
where
each
keypoint is unique
for each blob.

Print
the
number
of
coins
detected as well.

# Detect blobs

###

### YOUR CODE HERE

###

keypoints = detector.detect(finalImage)

9

# Print number of coins detected

###

### YOUR CODE HERE

###

print("Number of coins detect = ", len(keypoints))

​

Number
of
coins
detect = 9

Note
that
we
were
able
to
detect
all
9
coins.So, that
's your benchmark.
Step
4.3: Display
the
detected
coins
on
original
image

Make
sure
to
mark
the
center
of
the
blobs as well.Use
only
the
functions
discussed in Image
Annotation
section in Week
1
Hints

You
can
extract
the
coordinates
of
the
center and the
diameter
of
a
blob
using
k.pt and k.size
where
k is a
keypoint.

# Mark coins using image annotation concepts we have studied so far

###

### YOUR CODE HERE

###

finalImageBGR = cv2.cvtColor(finalImage, cv2.COLOR_GRAY2BGR)

​

for k in keypoints:
    x, y = k.pt

    x = int(round(x))

    y = int(round(y))

    cv2.circle(finalImageBGR, (x, y), 5, (255, 0, 0), 1)

    # Get radius of blob

    diameter = k.size

    radius = int(round(diameter / 2))

    # Mark blob in RED

    cv2.circle(finalImageBGR, (x, y), radius, (0, 0, 255), 2)

    # Annotate original Image

    cv2.circle(image, (x, y), 5, (255, 0, 0), 1)

    cv2.circle(image, (x, y), radius, (0, 255, 0), 2)

​

​

plt.imshow(finalImageBGR)

< matplotlib.image.AxesImage
at
0x7fd9936354e0 >

# Display the final image

###

### YOUR CODE HERE

###

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

plt.title("Final Annotated Image")

​

Text(0.5, 1, 'Final Annotated Image')

Step
4.4: Perform
Connected
Component
Analysis