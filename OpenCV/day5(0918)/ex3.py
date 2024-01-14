import cv2
import sys

src = cv2.imread('soccer.jpg', cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed")
    sys.exit()
sift = cv2.SIFT.create()
kepoints, desc = sift.detectAndCompute(src, None)
dst = cv2.drawKeypoints(src, kepoints, None, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()

