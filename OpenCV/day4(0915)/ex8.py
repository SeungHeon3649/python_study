# 도형 찾기

# import skimage 
# import cv2
# import numpy as np
# import sys

# def setLabel(src, pts, label):
#     (x, y, w, h) = cv2.boundingRect(pts)
#     pt1 = (x, y)
#     pt2 = (x + w, y + h)
#     cv2.rectangle(src, pt1, pt2, (255, 0, 0), 2)
#     cv2.putText(src, label, (pt1[0], pt1[1]), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

# src = cv2.imread('polygon.png', cv2.IMREAD_COLOR)
# if src is None:
#     print("Image load failed")
#     sys.exit()
# gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# #cv2.drawContours(dst, contours, -1, (255, 0, 0), 3)

# for c in contours:
#     approx = cv2.approxPolyDP(c, cv2.arcLength(c, True) * 0.02, True)
#     vtc = len(approx)
#     print(vtc)
#     if vtc == 3:
#         setLabel(src, c, "Triangle")
#     elif vtc == 4:
#         setLabel(src, c, "Rectangle")
#     elif vtc == 5:
#         setLabel(src, c, "Fiveangle")
#     else:
#         area = cv2.contourArea(c)
#         p, r = cv2.minEnclosingCircle(c)
#         ratio = r * r * 3.14 / area
#         if int(ratio) == 1:
#             setLabel(src, c, "Circle")
#         else:
#             setLabel(src, c, "Star")
# cv2.imshow("src", src)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 실습1(오브젝트 하나씩)

# import cv2
# import numpy as np
# img=cv2.imread('soccer.jpg')
# im_sw = np.copy(img)

# #cv2.GC_BGD#확정 배경
# #cv2.GC_FGD#확정 물체
# #cv2.GC_PR_BGD#배경 일꺼야
# #cv2.GC_PR_FGD#물체 일꺼야

# mask_im=np.zeros((im_sw.shape[0],im_sw.shape[1]),np.uint8)
# mask_im[:,:]=cv2.GC_PR_BGD

# def f(event,x,y,f,p):
#     if event==cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(im_sw,(x,y),10,(0,0,255),-1)
#         cv2.circle(mask_im,(x,y),10,cv2.GC_FGD,-1)
                   
#     elif event==cv2.EVENT_RBUTTONDOWN:
#         cv2.circle(im_sw,(x,y),10,(255,0,0),-1)
#         cv2.circle(mask_im,(x,y),10,cv2.GC_BGD,-1)

#     elif event==cv2.EVENT_MOUSEMOVE and f == cv2.EVENT_FLAG_LBUTTON:
#         cv2.circle(im_sw,(x,y),10,(0,0,255),-1)
#         cv2.circle(mask_im,(x,y),10,cv2.GC_FGD,-1)

#     elif event==cv2.EVENT_MOUSEMOVE and f == cv2.EVENT_FLAG_RBUTTON: 
#         cv2.circle(im_sw,(x,y),10,(255,0,0),-1)
#         cv2.circle(mask_im,(x,y),10,cv2.GC_BGD,-1)

#     cv2.imshow("main",im_sw)

# cv2.namedWindow("main")
# cv2.setMouseCallback('main',f)
# while True:
#     if cv2.waitKey(1)==27:
#         break

# f_h=np.zeros((1,65),np.float64)#물체
# b_h=np.zeros((1,65),np.float64)#배경

# cv2.grabCut(img,mask_im,None,b_h,f_h,5,cv2.GC_INIT_WITH_MASK)
# mask2=np.where((mask_im==cv2.GC_BGD)|(mask_im==cv2.GC_PR_BGD),0,1).astype('uint8')
# grab_img=img*mask2[:,:,np.newaxis]
# r,thimg=cv2.threshold(cv2.cvtColor(grab_img,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY)

# # ck_img = 255 - np.uint8(thimg * 255)
# c, h = cv2.findContours(thimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# img2 = cv2.cvtColor(thimg, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(img2, c, -1, (255, 0, 0), 3)
# cv2.imshow('w1', img2)


# img3 = cv2.cvtColor(thimg, cv2.COLOR_GRAY2BGR)

# for i in range(len(c)):
#     c_out = c[i]
#     # 직선 근사
#     c_a = cv2.approxPolyDP(c_out, 20, True)

#     cv2.drawContours(img3, [c_a], -1, (255, 0, 0), 3)

# # 영역을 통해서 객체를 봄
#     hull = cv2.convexHull(c_out)
#     hull = hull.reshape(1, hull.shape[0], hull.shape[2])
#     cv2.drawContours(img3, hull, -1, (0, 0, 255), 3)
# cv2.imshow('w4', img3)
# cv2.waitKey()

# 실습2(오브젝트 하나로)

import cv2
import numpy as np
img=cv2.imread('soccer.jpg')
im_sw = np.copy(img)

#cv2.GC_BGD#확정 배경
#cv2.GC_FGD#확정 물체
#cv2.GC_PR_BGD#배경 일꺼야
#cv2.GC_PR_FGD#물체 일꺼야

mask_im=np.zeros((im_sw.shape[0],im_sw.shape[1]),np.uint8)
mask_im[:,:]=cv2.GC_PR_BGD

def f(event,x,y,f,p):
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im_sw,(x,y),10,(0,0,255),-1)
        cv2.circle(mask_im,(x,y),10,cv2.GC_FGD,-1)
                   
    elif event==cv2.EVENT_RBUTTONDOWN:
        cv2.circle(im_sw,(x,y),10,(255,0,0),-1)
        cv2.circle(mask_im,(x,y),10,cv2.GC_BGD,-1)

    elif event==cv2.EVENT_MOUSEMOVE and f == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(im_sw,(x,y),10,(0,0,255),-1)
        cv2.circle(mask_im,(x,y),10,cv2.GC_FGD,-1)

    elif event==cv2.EVENT_MOUSEMOVE and f == cv2.EVENT_FLAG_RBUTTON: 
        cv2.circle(im_sw,(x,y),10,(255,0,0),-1)
        cv2.circle(mask_im,(x,y),10,cv2.GC_BGD,-1)

    cv2.imshow("main",im_sw)

cv2.namedWindow("main")
cv2.setMouseCallback('main',f)
while True:
    if cv2.waitKey(1)==27:
        break

f_h=np.zeros((1,65),np.float64)#물체
b_h=np.zeros((1,65),np.float64)#배경

cv2.grabCut(img,mask_im,None,b_h,f_h,5,cv2.GC_INIT_WITH_MASK)
mask2=np.where((mask_im==cv2.GC_BGD)|(mask_im==cv2.GC_PR_BGD),0,1).astype('uint8')
grab_img=img*mask2[:,:,np.newaxis]
r,thimg=cv2.threshold(cv2.cvtColor(grab_img,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY)

# ck_img = 255 - np.uint8(thimg * 255)
c, h = cv2.findContours(thimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#img2 = cv2.cvtColor(thimg, cv2.COLOR_GRAY2BGR)
#cv2.drawContours(img2, c, -1, (255, 0, 0), 3)
#cv2.imshow('w1', img2)

img3 = cv2.cvtColor(thimg, cv2.COLOR_GRAY2BGR)
all_contours = np.concatenate(c)
print(all_contours)
cv2.drawContours(img3, [all_contours], -1, (0, 0, 255), 3)
hull = cv2.convexHull(all_contours)
hull = hull.reshape(1, hull.shape[0], hull.shape[2])
cv2.drawContours(img3, hull, -1, (0, 255, 0), 3)
cv2.imshow('w4', img3)
cv2.waitKey()
cv2.destroyAllWindows()
