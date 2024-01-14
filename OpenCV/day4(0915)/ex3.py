import cv2
import sys
img1 = cv2.imread('d3.jpg')
if img1 is None:
    print("Image load failed")
    sys.exit()
gr_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
c1=cv2.Canny(gr_img1,100,200)

c,h=cv2.findContours(c1,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
print(len(c))

l=[]
for i in range(len(c)):
    if c[i].shape[0]>100:
        l.append(c[i])
print(len(l))
cv2.drawContours(img1,l,-1,(255,0,0),3)
cv2.imshow('img',img1)
#cv2.drawContours(img1,c,-1,(255,0,0),3)
#cv2.imshow('img',img1)
cv2.waitKey()