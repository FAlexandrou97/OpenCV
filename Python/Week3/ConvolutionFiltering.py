import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("musk.jpg")

kernel_size = 3
# Create a 5*5 kernel with all elements equal to 1
for i in range(10):
    kernel = np.random.rand(kernel_size, kernel_size)

    # Print Kernel
    print(kernel)

    result = cv2.filter2D(img, -1, kernel, (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)

    plt.figure(figsize=[10,5])
    plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Original Image")
    plt.subplot(122);plt.imshow(result[...,::-1]);plt.title("Convolution Result")
    plt.waitforbuttonpress()

