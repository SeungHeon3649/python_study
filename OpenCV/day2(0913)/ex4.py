import cv2
import sys

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
r = (0, 0, 255)
g = (0, 255, 0)
ck = [r, g]
def f(event, x, y, flags, param):
    global ck
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(src, (x, y), 5, ck[0], -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        ck = ck[::-1]
    cv2.imshow("src", src)
# cv2.circle(src, (100, 100), 5, (0, 0, 255), -1)
# cv2.circle(src, (200, 200), 5, (0, 0, 255), -1)
# cv2.circle(src, (300, 300), 5, (0, 0, 255), -1)
cv2.imshow("src", src)
cv2.setMouseCallback("src", f)
cv2.waitKey()
cv2.destroyAllWindows()