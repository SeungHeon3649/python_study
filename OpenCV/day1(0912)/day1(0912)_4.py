# 저장된 동영상 불러오기

import cv2
import numpy as np
import pandas as pd
import sys

src = cv2.VideoCapture("m_v.avi")
if src.isOpened() == False:
    print("동영상 안불러와짐")
    sys.exit()

while True:
    ret, frame = src.read()
    if ret == False:
        print("동영상 출력 완료")
        break
    cv2.imshow("mv", frame)
    if cv2.waitKey(33) == 27:
        break
src.release()
cv2.destroyAllWindows()