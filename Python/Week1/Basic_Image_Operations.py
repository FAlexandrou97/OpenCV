# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Press any key to load next image

# Load Image using matploitlib
img = plt.imread("test.jpg")
# Convert BGR to RGB color
# Needed e.g when loading image through openCV and showing it through matploitlib
imgPLT = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.interactive(True)
plt.imshow(img)
plt.waitforbuttonpress(0)
plt.close()

# Load Image
img = cv2.imread("test.jpg")
cv2.imshow("openCV", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw a black rectangle
emptyMatrixBlack = np.zeros((200, 200, 3), dtype='uint8')
cv2.imshow("empty matrix Black", emptyMatrixBlack)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw a white rectangle
emptyMatrixWhite = 255 * np.ones((200, 200, 3), dtype='uint8')
cv2.imshow("empty matrix White", emptyMatrixWhite)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Crop and show Image
crop = img[:250, :300]
cv2.imshow("Cropped Image", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Copy Elon's face and paste it
elon = cv2.imread("musk.jpg")
elonCopy = elon.copy()
copyRoi = elon[20:135, 100:190]
roiHeight, roiWidth = copyRoi.shape[:2]
# Copy to left of Face
elonCopy[40:40+roiHeight, 10:10+roiWidth] = copyRoi
# Copy to right of Face
elonCopy[40:40+roiHeight, 200:200+roiWidth] = copyRoi
# Display images in a plt interactive window
plt.figure(figsize=[15, 15])
plt.subplot(121);plt.imshow(elon[..., ::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(elonCopy[..., ::-1]);plt.title("Output Image")
plt.waitforbuttonpress(0)
plt.close()

# Resize the image by specifically assigning its width and height
resizeUpWidth = 600
resizeUpHeight = 900
resizedUp = cv2.resize(elon, (resizeUpWidth, resizeUpHeight), interpolation= cv2.INTER_LINEAR)

# Scaling the image 10 times by specifying both scaling factors
# Might make the image clearer
scaleUpX = 10
scaleUpY = 10
scaledUp = cv2.resize(elon, None, fx= scaleUpX, fy= scaleUpY, interpolation= cv2.INTER_LINEAR)
# Display images in a plt interactive window
plt.figure(figsize=[15, 15])
plt.subplot(131);plt.imshow(elon[:,:,::-1]);plt.title("Original Image")
plt.subplot(132);plt.imshow(resizedUp[:,:,::-1]);plt.title("Scaled Up Image Using Specific Width And Height")
plt.subplot(133);plt.imshow(scaledUp[:,:,::-1]);plt.title("Scaled Up Image Using Scaling Factor")
plt.waitforbuttonpress(0)
plt.close()

mask = np.zeros_like(elon)
mask[20:135, 100:190] = 255
mask2 = cv2.inRange(elon, (0, 0, 150), (100, 100, 255))
# Display images in a plt interactive window
plt.figure(figsize=[15, 15])
plt.subplot(131);plt.imshow(elon[:,:,::-1]);plt.title("Original Image")
plt.subplot(132);plt.imshow(mask[:,:,::-1]);plt.title("Mask")
plt.subplot(133);plt.imshow(mask2);plt.title("Mask")
plt.waitforbuttonpress(0)
plt.close()
