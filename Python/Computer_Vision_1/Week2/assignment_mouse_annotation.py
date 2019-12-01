import cv2


def drawRect(action, x, y, flags, userdata):
    # Referencing global variables 
    global center, circumference, rectangle, buttonPressed
    # Action to be taken when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        buttonPressed = True
        startPos = [(x, y)]
        rectangle.append(startPos)
        # Draw the starting dot
        cv2.rectangle(source, pt1=startPos[0], pt2=startPos[0], color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # Action to be taken when left mouse button is released
    elif action == cv2.EVENT_LBUTTONUP:
        buttonPressed = False
        endPos = [(x, y)]
        rectangle.append(endPos)
        # Draw the rectangle
        cv2.rectangle(source, rectangle[0][0], rectangle[1][0], (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Window", source)

        # Save the cropped image
        crop = source[rectangle[0][0][1]:rectangle[1][0][1], rectangle[0][0][0]:rectangle[1][0][0]]
        cv2.imwrite("face.png", crop)

        rectangle = []


# Lists to store the points
center = []
circumference = []
rectangle = []
buttonPressed = False

source = cv2.imread("sample.jpg", 1)
# Make a dummy image, will be useful to clear the drawing
dummy = source.copy()

k = 0
# loop until escape character is pressed
while k != 27:

    # highgui function called when mouse events occur
    cv2.setMouseCallback("Window", drawRect)

    cv2.imshow("Window", source)
    cv2.putText(source, '''Choose center, and drag, 
                      Press ESC to exit and c to clear''',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    k = cv2.waitKey(20) & 0xFF
    # Another way of cloning
    if k == 99:
        source = dummy.copy()

cv2.destroyAllWindows()
