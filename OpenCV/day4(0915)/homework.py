import cv2
import numpy as np
import sys

src = cv2.VideoCapture(0)
if src.isOpened() == False:
    print("camera load failed")
    sys.exit()

while True:
    ret, frame = src.read()
    if ret == False:
        print("캡쳐불가")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    contours, h = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

src.release()
cv2.destroyAllWindows()