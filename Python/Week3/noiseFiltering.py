import cv2
import matplotlib.pyplot as plt

# Load Images
saltPepper = cv2.imread('images/SaltPepperNoise.png')
gaussian = cv2.imread('images/gaussian_noise.png')

# Convert to rgb
saltPepper = cv2.cvtColor(saltPepper, cv2.COLOR_BGR2RGB)
gaussian = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)

# Salt and Pepper noise filtering
saltPepperMedianFiltering = cv2.medianBlur(saltPepper,3)
saltPepperBilateralFiltering = cv2.bilateralFilter(saltPepper,d=10, sigmaColor=200, sigmaSpace=80)

plt.figure(figsize=[10,8])
plt.subplot(221);plt.imshow(saltPepper);plt.title('Salt n Pepper Noise')
plt.subplot(222);plt.imshow(saltPepperMedianFiltering);plt.title('Median Filtering')
plt.subplot(223);plt.imshow(saltPepperBilateralFiltering);plt.title('Bilateral Filtering')
plt.waitforbuttonpress()
plt.close()

# Gaussian Noise filtering
gaussianMedianFiltering = cv2.medianBlur(gaussian, 7)
gaussianBilateralFiltering = cv2.bilateralFilter(gaussian, d=15, sigmaColor=70, sigmaSpace=40)

plt.figure(figsize=[10,8])
plt.subplot(221);plt.imshow(gaussian);plt.title('Gaussian Noise')
plt.subplot(222);plt.imshow(gaussianMedianFiltering);plt.title('Median Filtering')
plt.subplot(223);plt.imshow(gaussianBilateralFiltering);plt.title('Bilateral Filtering')
plt.waitforbuttonpress()
plt.close()

# Gaussian Noise excessive filtering
gHeavyMedianFiltering = cv2.medianBlur(gaussian, 13)
gHeavyBilateralFiltering = cv2.bilateralFilter(gaussian, d=20, sigmaColor=100, sigmaSpace=40)

plt.figure(figsize=[10,8])
plt.subplot(221);plt.imshow(gaussian);plt.title('Gaussian Noise')
plt.subplot(222);plt.imshow(gHeavyMedianFiltering);plt.title('Heavy Median Filtering')
plt.subplot(223);plt.imshow(gHeavyBilateralFiltering);plt.title('Heavy Bilateral Filtering')
plt.waitforbuttonpress()
