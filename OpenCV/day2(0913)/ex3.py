# 마우스 이벤트

import cv2
import sys

def draw(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(src, (ix, iy), (x, y), (0, 0, 255), 5)
    cv2.imshow("src", src)

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
cv2.imshow("src", src)
cv2.setMouseCallback("src", draw)
cv2.waitKey()
cv2.destroyAllWindows()
