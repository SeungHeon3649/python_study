import cv2
import sys
import matplotlib.pyplot as plt

src = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
if src is None:
    print("Image load failed")
    sys.exit()

hb = cv2.calcHist([src], [0], None, [256], [0, 256])
hg = cv2.calcHist([src], [1], None, [256], [0, 256])
hr = cv2.calcHist([src], [2], None, [256], [0, 256])
print(src.shape)
r_src = src[:, :, 2]
g_src = src[:, :, 1]
b_src = src[:, :, 0]
f_rg = r_src.reshape(-1)

plt.plot(hb, 'b-')
plt.plot(hg, 'g-')
plt.plot(hr, 'r-')    
plt.show()
plt.hist(f_rg)
plt.show()