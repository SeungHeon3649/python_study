import cv2
import sys
import numpy as np

src = cv2.imread('apple.webp', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
circle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1 = 150,
                  param2 = 20, minRadius = 50, maxRadius = 200)

for i in circle[0]:
    cv2.circle(src, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 3)

cv2.imshow("src", src)
cv2.waitKey()
cv2.destroyAllWindows()