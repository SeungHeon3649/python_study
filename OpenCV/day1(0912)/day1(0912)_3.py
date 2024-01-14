# 동영상 저장

import cv2
import numpy as np
import pandas as pd
import sys

src = cv2.VideoCapture(0)
if src.isOpened() == False:
    print("카메라 연결 안됨")
    sys.exit()
ret, frame = src.read()
if ret == False:
    print("캡쳐 불가")
    exit()

codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 30.0
#print(frame.shape[:2])
h, w = frame.shape[:2]
m_v = cv2.VideoWriter("m_v.avi", codec, fps, (w,h))

if m_v.isOpened() == False:
    print("동영상을 만들수 없습니다")
    sys.exit()

# width = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(width)
# print(height)

while True:
    ret, frame = src.read()
    if ret == False:
        print("캡쳐불가")
        break
    m_v.write(frame)
    cv2.imshow("m_v", frame)
    key = cv2.waitKey(33)
    if key == 27:
        break
m_v.release()
src.release()
cv2.destroyAllWindows()

