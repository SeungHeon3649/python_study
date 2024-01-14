import cv2
import sys
import numpy as np

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()

#ret, thresh = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
canny = cv2.Canny(src, 100, 200)
c, h = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

l = []
for i in range(len(c)):
    if c[i].shape[0] > 100:
        l.append(c[i])

cv2.drawContours(src, l, -1, (255, 0, 0), 3)
cv2.imshow("src", src)
#cv2.imshow("thresh", thresh)
cv2.imshow("canny", canny)
cv2.waitKey()
cv2.destroyAllWindows()