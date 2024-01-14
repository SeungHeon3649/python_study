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
cv2.imshow("out_im",thimg)

cv2.waitKey()
cv2.destroyAllWindows()