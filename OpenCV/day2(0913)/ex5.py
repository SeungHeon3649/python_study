# 마우스 이벤트로 사각형 그리기

import cv2
import sys

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
start_x, start_y, end_x, end_y = -1, -1, -1, -1
def cut(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y
    global dst
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x, y
        #print(end_x, end_y)
        cv2.rectangle(src, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
        #print(end_x, end_y)
        dst = src[start_y:end_y, start_x:end_x]
        cv2.imshow("dst", dst)
    #print(start_x, start_y, end_x, end_y)
    #print(roi)
    cv2.imshow("src", src)
cv2.imshow("src", src)
cv2.setMouseCallback("src", cut)
cv2.waitKey()
cv2.destroyAllWindows()
