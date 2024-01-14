# 이미지 불러오기

import cv2
import sys
import matplotlib.pyplot as plt

img = cv2.imread('cat.bmp', cv2.IMREAD_COLOR)
if img is None:
    print('Image load failed')
    sys.exit()
#cv2.namedWindow('cat')
cv2.imshow('cat', img)
cv2.waitKey()
cv2.destroyAllWindows()

dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(dst)
plt.show()