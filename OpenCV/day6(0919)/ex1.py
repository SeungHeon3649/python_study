import cv2
import numpy as np
import sys

def load_img(*img_names):
    end_d=[]
    for img_name in img_names:
        img=cv2.imread(img_name)
        gr_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        end_d.append((img,gr_img))
    return end_d
d=load_img('mot_color70.jpg',"mot_color83.jpg")
old_im=d[0][0][190:350, 440:560]
gr_old=d[0][1][190:350, 440:560]
new_im1=d[1][0]
gr_new1=d[1][1]

sift=cv2.SIFT_create()
old_kp,old_des=sift.detectAndCompute(gr_old,None)
new_kp,new_des=sift.detectAndCompute(gr_new1,None)

flann_matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matcher1 = flann_matcher.knnMatch(old_des,new_des,2)

T=0.8
m_l=[old_des for old_des,new_des in knn_matcher1 if old_des.distance/new_des.distance<T]

point_1 = np.float32([old_kp[gm.queryIdx].pt for gm in m_l])
point_2 = np.float32([new_kp[gm.trainIdx].pt for gm in m_l])

H, _ = cv2.findHomography(point_1, point_2, cv2.RANSAC)

h1, w1 = old_im.shape[0], old_im.shape[1]
h2, w2 = new_im1.shape[0], new_im1.shape[1]

box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(4, 1, 2)
box2 = cv2.perspectiveTransform(box1, H)

new_im1 = cv2.polylines(new_im1, [np.int32(box2)], True, (0, 255, 0), 8)
mc_img1 = np.empty((max(old_im.shape[0], new_im1.shape[0]),
                    old_im.shape[1] + new_im1.shape[1], 3), np.uint8)
cv2.drawMatches(old_im, old_kp, new_im1, new_kp, m_l, mc_img1,
                flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow(f"end1_im", mc_img1)

cv2.waitKey()
cv2.destroyAllWindows()
