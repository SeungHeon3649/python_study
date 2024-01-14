import numpy as np
import cv2

data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  dtype = np.float32)
ux = np.array([[-1, 0, 1]])   # y 방향 마스크(수평마스크)
uy = np.array([[-1, 0, 1]]).T # x 방향 마스크(수직마스크)
k = cv2.getGaussianKernel(3, 1)
g = np.outer(k, k.T)
dy = cv2.filter2D(data, cv2.CV_32F, uy)
dx = cv2.filter2D(data, cv2.CV_32F, ux)

dyy = dy * dy
dxx = dx * dx
dyx = dy * dx

gdyy = cv2.filter2D(dyy, cv2.CV_32F, g)
gdxx = cv2.filter2D(dxx, cv2.CV_32F, g)
gdyx = cv2.filter2D(dyx, cv2.CV_32F, g)

C = (gdyy * gdxx - gdyx * gdyx) - 0.04 * (gdyy + gdxx) * (gdyy + gdxx)
print(C[4, 2])


for i in range(1, C.shape[0] - 1):
    for j in range(1, C.shape[1] - 1):
        print(C[j, i], j, i)
