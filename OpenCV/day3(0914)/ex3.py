# 그레이스케일 시간복잡도 테스트

import cv2
import numpy as np
import time
import sys

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()

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

st = time.time()
gr_im1 = gr_f1(src)
end = time.time()
print("1번 : ", end - st)

st = time.time()
gr_im2 = gr_f2(src)
end = time.time()
print("2번 : ", end - st)

st = time.time()
gr_im3 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
end = time.time()
print("3번 : ", end - st)

cv2.imshow("gr_im1", gr_im1)
cv2.imshow("gr_im2", gr_im2)
cv2.imshow("gr_im3", gr_im3)
cv2.waitKey()
cv2.destroyAllWindows()