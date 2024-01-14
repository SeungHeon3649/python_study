import skimage 
import cv2
import numpy as np

img=skimage.data.horse()
ck_img = 255 - np.uint8(img) * 255
cv2.imshow('t',ck_img)
c, h = cv2.findContours(ck_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
img2 = cv2.cvtColor(ck_img,cv2.COLOR_GRAY2BGR)
cv2.drawContours(img2, c, -1, (255,0,0), 3)
cv2.imshow('img2', img2)

c_out=c[0]
o_m=cv2.moments(c_out)
area=cv2.contourArea(c_out) # 면적
c_x,c_y = o_m['m10'] / o_m['m00'], o_m['m01'] / o_m['m00'] # 중점
# m00 : 면적, m10 : x좌표의 합, m01 : y좌표의 합
p=cv2.arcLength(c_out,True)#둘레
print(f'면적{area},중점({c_x},{c_y}),둘래{p}')

img3 = cv2.cvtColor(ck_img, cv2.COLOR_GRAY2BGR)
c_a = cv2.approxPolyDP(c_out, 8, True)
print(c_a[10])
cv2.drawContours(img3, [c_a], -1, (255, 0, 0), 3)

hull = cv2.convexHull(c_out)
hull = hull.reshape(1, hull.shape[0], hull.shape[2])
cv2.drawContours(img3, hull, -1, (0, 0, 255), 3)
cv2.imshow("img3", img3)

cv2.waitKey()
