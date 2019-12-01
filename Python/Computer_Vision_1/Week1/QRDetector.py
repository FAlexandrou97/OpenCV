# Import modules
import cv2
import matplotlib.pyplot as plt

# STEP 1
# Read image
img = cv2.imread("IDCard-Satya.png")

# STEP 2
# Create a QRCodeDetector Object
# Variable name should be qrDecoder

qrDecoder = cv2.QRCodeDetector()

# Detect QR Code in the Image
# Output should be stored in
# opencvData, bbox, rectifiedImage
# in the same order

opencvData, bbox, rectifiedImage = qrDecoder.detectAndDecode(img)

# Check if a QR Code has been detected
if opencvData != None:
    print("QR Code Detected")
else:
    print("QR Code NOT Detected")

# STEP 3
n = len(bbox)
# Draw the bounding box
for i in range(1, n):
    cv2.line(img, pt1=(bbox[i-1][0, 0], bbox[i-1][0, 1]), pt2=(bbox[i][0, 0], bbox[i][0, 1]),
             color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # Draw last line
    if i == n-1:
        cv2.line(img, pt1=(bbox[i][0, 0], bbox[i][0, 1]), pt2=(bbox[0][0, 0], bbox[0][0, 1]),
                 color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

# STEP 4
# Since we have already detected and decoded the QR Code
# using qrDecoder.detectAndDecode, we will directly
# use the decoded text we obtained at that step (opencvdata)

print("QR Code Detected!")
print(opencvData)

# STEP 5
# Write the result image
resultImagePath = "QRCode-Output.png"
cv2.imwrite(resultImagePath, img)
img = cv2.imread(resultImagePath)
# Display the result image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.interactive(True)
plt.imshow(img)

cv2.imshow("NEW", img)
cv2.waitKey(0)
