# Combination of real-time object detection and tracking
Used [YOLOv3 from OpenCV DNN module](https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#gafde362956af949cce087f3f25c6aff0d) for detection
and [KCF Tracker](https://docs.opencv.org/3.4/d2/dff/classcv_1_1TrackerKCF.html) for tracking.

Input video is of a football (soccer) game, and the resized captured frames consist of 416x416 pixels.

The aim of the application is to constantly keep track of the ball while maintaining a relatively high framerate.

Requires [YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights) to run!

### Output frame during tracking
![image](https://i.ibb.co/WBSF6yk/1.png)
### Output frame during detection
![image](https://i.ibb.co/gjb4Y1w/2.png)
