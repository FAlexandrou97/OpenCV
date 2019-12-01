import cv2
import numpy as np

image = cv2.imread('hillary_clinton.jpg')
# Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.03, 5)
haarPrediction = image.copy()
for (x, w, y, h) in face:
    cv2.rectangle(haarPrediction, (x, y), (x+w, y+h), (255, 0, 0) , 2)

cv2.imshow('haarCascade Detection', haarPrediction)
cv2.waitKey(0)

# Tensorflow model face detector
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
blob = cv2.dnn.blobFromImage(image=image,scalefactor=1, size=(300, 300), mean=0, swapRB=True, crop=0, ddepth=0)
net.setInput(blob)
detections = net.forward()
bboxes = []
tfPrediction = image.copy()
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.9:
        x1 = int(detections[0, 0, i, 3] * image.shape[1])
        y1 = int(detections[0, 0, i, 4] * image.shape[0])
        x2 = int(detections[0, 0, i, 5] * image.shape[1])
        y2 = int(detections[0, 0, i, 6] * image.shape[0])
        bboxes.append([x1, y1, x2, y2])
        cv2.rectangle(tfPrediction, (x1, y1), (x2, y2), (0, 255, 0), int(round(image.shape[0] / 150)), 8)
        cropped_face = image[y1:y2, x1:x2]
        # Would normally use non-maximum suppression but code is kept small for the sake of the project
        break

cv2.imshow('tensorflow model Detection', tfPrediction)
cv2.waitKey(0)
cv2.imshow('Cropped Face', cropped_face)
cv2.waitKey(0)
