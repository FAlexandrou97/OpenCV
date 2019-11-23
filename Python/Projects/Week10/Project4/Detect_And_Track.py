import numpy as np
import cv2
import time


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold and classId == 32:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return boxes

# START OF MAIN PROGRAM
# Initialize the parameters
objectnessThreshold = 0.5 # Objectness threshold
confThreshold = 0.5       # Confidence threshold
nmsThreshold = 0.4        # Non-maximum suppression threshold
inpWidth = 416            # Width of network's input image
inpHeight = 416           # Height of network's input image

classesFiles = "coco.names"
classes = None
with open(classesFiles, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

# Read model
net = cv2.dnn.readNetFromDarknet(cfgFile=modelConfig, darknetModel=modelWeights)
cap = cv2.VideoCapture('soccer-ball.mp4')
startTime = time.time()
startDetectionTime = time.time()
detectionTime = 0
tracker = cv2.TrackerKCF_create()
trackerInit = True
allowDetect = True
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Detect Object
        if allowDetect:
            print(detectionTime)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
            net.setInput(blob)
            outs = net.forward(getOutputsNames(net))
            box = postprocess(frame, outs)
            if any(x is not 0 for x in box) and len(box) == 1:
                trackerBox = tuple(sum(box, []))
            # print("DetectorBox: " , box)
            allowDetect = False
            tracker = cv2.TrackerKCF_create()
            ok = tracker.init(frame, trackerBox)
            cv2.putText(frame, "Detecting Object!", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            detectionTime = time.time() - startDetectionTime
            # print(detectionTime)

        # Initialise Tracker
        if trackerInit:
            ok = tracker.init(frame, (568, 323, 108, 102))
            trackerInit = False

        # print("trackerBox: " , trackerBox)
        if ok:
            # Tracking success
            p1 = (int(trackerBox[0]), int(trackerBox[1]))
            p2 = (int(trackerBox[0] + trackerBox[2]), int(trackerBox[1] + trackerBox[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
            startDetectionTime = time.time()
            ok, trackerBox = tracker.update(frame)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            allowDetect = True
            # Detect for a maximum of 1.5 seconds
            if detectionTime > 1.5:
                allowDetect = False
                detectionTime = time.time() - startDetectionTime
                # Pause detection for 1.5 second
                if detectionTime > 3:
                    startDetectionTime = time.time()
                    detectionTime = 0
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
