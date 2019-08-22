import cv2
import matplotlib.pyplot as plt

# Opening (Erosion and then Dilation) is used to clear small white spots
# Closing (Dilation and then Erosion) is used to clear small black spots

image = cv2.imread("opening.png", cv2.IMREAD_GRAYSCALE)

# Specify Kernel Size
kernelSize = 10

# Create the Kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernelSize+1, 2*kernelSize+1),
                                    (kernelSize, kernelSize))

# Perform Erosion
imEroded = cv2.erode(image, element, iterations=1)
# Perform Dilation
imOpen = cv2.dilate(imEroded, element, iterations=1)

# Display Kernel
plt.imshow(element, cmap='gray');plt.title("Structuring Element : Ellipse, Kernel Size : {}".format(kernelSize))
plt.waitforbuttonpress(0)
plt.close()

# Display Output
plt.figure(figsize=[15,15])
plt.subplot(131);plt.imshow(image, cmap='gray');plt.title("Original Image")
plt.subplot(132);plt.imshow(imEroded,cmap='gray');plt.title("After Erosion Operation")
plt.subplot(133);plt.imshow(imOpen,cmap='gray');plt.title("After Dilation Operation");
plt.waitforbuttonpress(0)
plt.close()

# Get structuring element/kernel which will be used
# for opening operation
openingSize = 3

# Selecting a elliptical kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (2 * openingSize + 1, 2 * openingSize + 1), (openingSize, openingSize))

imageMorphOpened = cv2.morphologyEx(image, cv2.MORPH_OPEN,
                                    element, iterations=3)
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image, cmap='gray');plt.title("Original Image")
plt.subplot(122);plt.imshow(imageMorphOpened, cmap='gray');plt.title("After Opening Operation")
plt.waitforbuttonpress(0)
plt.close()

image = cv2.imread("closing.png", cv2.IMREAD_GRAYSCALE)

# Specify Kernel Size
kernelSize = 10

# Create Kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernelSize+1, 2*kernelSize+1),
                                    (kernelSize, kernelSize))

# Perform Dilation
imDilated = cv2.dilate(image, element)
# Perform Erosion
imClose = cv2.erode(imDilated, element)

plt.figure(figsize=[15,15])
plt.subplot(131);plt.imshow(image, cmap='gray');plt.title("Original Image")
plt.subplot(132);plt.imshow(imDilated,cmap='gray');plt.title("After Dilation Operation")
plt.subplot(133);plt.imshow(imClose,cmap='gray');plt.title("After Erosion Operation")
plt.waitforbuttonpress(0)
plt.close()

# Get structuring element/kernel
# which will be used for closing operation
closingSize = 10

# Selecting an elliptical kernel
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (2 * closingSize + 1, 2 * closingSize + 1), (closingSize, closingSize))

imageMorphClosed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image, cmap='gray');plt.title("Original Image")
plt.subplot(122);plt.imshow(imageMorphClosed, cmap='gray');plt.title("After Closing Operation")
plt.waitforbuttonpress(0)
plt.close()

