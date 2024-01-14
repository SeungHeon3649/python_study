import cv2
import numpy as np

c_imgs = cv2.VideoCapture("v1.mp4")
c_imgs = cv2.VideoCapture(0)

f_params = dict(maxCorners = 100, qualityLevel = 0.3,
                minDistance = 7, blockSize = 7)
lk_params = dict(winSize = (15, 15), maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

r, old_frame = c_imgs.read()
old_gr = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gr, mask = None, **f_params)
mask = np.zeros_like(old_frame)

while True:
    ret, frame = c_imgs.read()
    if not ret: break

    new_gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, match, err = cv2. calcOpticalFlowPyrLK(old_gr, new_gr, p0, None,
                                               **lk_params)

    if p1 is not None:
        gd_new = p1[match == 1]
        gd_old = p0[match == 1]

    for i in range(len(gd_new)):
        a, b = int(gd_new[i][0]), int(gd_new[i][1])
        c, d = int(gd_old[i][0]), int(gd_old[i][1])
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)
    cv2.imshow("mc_img", img)
    key = cv2.waitKey(30)
    old_gray = new_gr.copy()
    if key == 27:
        break
    elif key == ord('c'):
        mask = np.zeros_like(frame)
        p0 = cv2.goodFeaturesToTrack(old_gr, mask=None, **f_params)

    #p0 = gd_new.reshape(-1, 1, 2)
cv2.destroyAllWindows()

