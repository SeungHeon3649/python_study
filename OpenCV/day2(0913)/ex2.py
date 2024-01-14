# 마우스 이벤트

import cv2
import sys

def draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(src, (x, y), (x + 100, y + 100), (0, 0, 255), 5)
    cv2.imshow("src", src)

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
cv2.imshow("src", src)
cv2.setMouseCallback("src", draw)
cv2.waitKey()
cv2.destroyAllWindows()
