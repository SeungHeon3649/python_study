# 카메라 켜기

import cv2
import numpy as np
import pandas as pd
import sys

src = cv2.VideoCapture(0)
if src.isOpened() == False:
    print("카메라 연결 안됨")
    sys.exit()
while True:
    ret, frame = src.read()
    if not ret: break
    cv2.imshow("camera", frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
src.release()
cv2.destroyAllWindows()