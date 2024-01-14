

import cv2
import numpy as np
import pandas as pd
import sys
import time

def gr_f1(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    for i in range(bgr_img.shape[0]):
        for j in range(bgr_img.shape[1]):
            g[i, j] = 0.114 * bgr_img[i, j, 0] + 0.587 * bgr_img[i, j, 0] + 0.299 * bgr_img[i, j, 0]
    return np.uint8(g)

def gr_f2(bgr_img):
    g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
    g = 0.114 * bgr_img[:, :, 0] + 0.587 * bgr_img[:, :, 0] + 0.299 * bgr_img[:, :, 0]
    return np.uint8(g)



src = cv2.VideoCapture(0)
if src.isOpened() == False:
    print("카메라 연결 안됨")
    sys.exit()

while True:
    ret, frame = src.read()
    if ret == False:
        print("캡쳐불가")
        break
    dst = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", dst)
    
    dst1 = cv2.GaussianBlur(dst, (15, 15), 0, 0)
    cv2.imshow("dst1", dst1)

    mask = np.array([[-1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 1]])
    g_img16 = np.int16(dst)
    dst2 = np.uint8(np.clip(cv2.filter2D(g_img16, -1, mask) + 128, 0, 255))
    cv2.imshow('dst2', dst2)

    key = cv2.waitKey(1)
    if key == 27:
        break
src.release()
cv2.destroyAllWindows()

