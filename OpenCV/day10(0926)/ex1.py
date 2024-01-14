import numpy as np
import cv2
import sys


def draw_f(img, flow, step = 16):
    for y in range(step // 2, img.shape[0], step):
        for x in range(step // 2, img.shape[1], step):
            dx, dy = flow[y, x].astype('int32')

            if (dx * dx + dy * dy) > 1:
                cv2.line(img, (x, y), (x + dx, y + dy), (0, 0, 255), 2)
            else:
                cv2.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2)
                
cam = cv2.VideoCapture(0)

if not cam.isOpened(): sys.exit('카메라 연결 실패')

pre = None

while True:
    r, frame = cam.read()
    if not r: sys.exit('프레임 획득 불가')
    if pre is None:
        pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        continue
    curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pre, curr, None, 0.2, 3, 15, 3, 5, 1.2, 0)

    draw_f(frame, flow)
    cv2.imshow('ck_flow', frame)
    pre = curr
    key = cv2.waitKey(1)

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
