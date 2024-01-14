# 이미지 감마 조절(화소)

import cv2
import sys
import numpy as np

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
dst = cv2.resize(src, (0,0), fx = 0.5, fy = 0.5)
print(dst.shape)

dst1 = dst - (30, 30, 30)
dst1[dst1 < 0] = 0
dst1 = dst1.astype('uint8')
dst2 = dst - 30
cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)

def g_tr(f,g=1.0):
    s_f = f/255.0
    return np.uint8(255*(s_f**g))
end_data = np.hstack((g_tr(dst,0.5),g_tr(dst,0.7),g_tr(dst,1.0),g_tr(dst,2.0),g_tr(dst,3.0)))
cv2.imshow("g_all_data", end_data)
cv2.waitKey()
cv2.destroyAllWindows()    