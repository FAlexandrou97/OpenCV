# Import libraries
import cv2

red = (0, 0, 255)
cyanish = (255, 255, 0)
green = (0, 255, 0)
blue = (255, 0, 0)

# In this training exercise, elon will be annotated with a line, circle, rectangle, elipse and text
elon = cv2.imread("musk.jpg")
elonAnnotated = elon.copy()

# Draw Line
cv2.line(elonAnnotated, pt1=(10, 20), pt2=(70, 60), color=red, lineType=cv2.LINE_AA)

# Draw Circle
cv2.circle(elonAnnotated, center=(171, 73), radius=10, color=cyanish, lineType=cv2.LINE_AA)
cv2.circle(elonAnnotated, center=(130, 73), radius=10, color=cyanish, thickness=-1, lineType=cv2.LINE_AA)

# Draw Ellipse
cv2.ellipse(elonAnnotated, center=(150, 80), axes=(80, 50), angle=45, startAngle=0, endAngle=360,
            color=green, thickness=2, lineType=cv2.LINE_AA)
cv2.ellipse(elonAnnotated, center=(150, 80), axes=(80, 50), angle=135, startAngle=0, endAngle=360,
            color=blue, thickness=2, lineType=cv2.LINE_AA)

# Draw Rectangle
cv2.rectangle(elonAnnotated, pt1=(100, 20), pt2=(200, 150), color=red, thickness=2, lineType=cv2.LINE_AA)

# Draw Text
cv2.putText(elonAnnotated, text="Elon OP", org=(0, 170),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), lineType=cv2.LINE_AA)

cv2.imshow("Annotations", elonAnnotated)
cv2.waitKey(0)
