# 평활화

import cv2
import sys
import matplotlib.pyplot as plt

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()
cv2.imshow("src", src)

dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow("dst", dst)
g_h = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.plot(g_h)
plt.show()

#평활화
e_img = cv2.equalizeHist(dst)
cv2.imshow("e_img", e_img)
e_h = cv2.calcHist([e_img], [0], None, [256], [0, 256])
plt.plot(e_h)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()