import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt

m = load_model("m.h5")

def reset():
    global img

    img = np.ones((200, 520, 3), dtype = np.uint8 ) * 255
    for i in range(5):
        cv2.rectangle(img, (10 + i * 100, 50), (10 + (i + 1) * 100, 150), (0, 0, 255))
    cv2.putText(img, 's : reset m : mk_imgshow c : ck_img q : quit',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

def mk_num():
    nums = []
    for i in range(5):
        ri = img[51:149, 11 + (i * 100): 9 + (i + 1) * 100]
        ri = 255 - cv2.resize(ri, (28, 28), interpolation = cv2.INTER_CUBIC)
        nums.append(ri)
    nums = np.array(nums)
    return nums

def mk_imgshow():
    nums = mk_num()
    plt.figure(figsize = (25, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(nums[i], cmap = 'gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

def ck_m():
    nums = mk_num()
    s_x = nums.reshape(-1, 28, 28, 1) / 255.0
    y = m.predict(s_x)
    class_l = np.argmax(y, axis = 1)
    for i in range(5):
        cv2.putText(img, str(class_l[i]), (50 + i * 100, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)


def writing(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0, 0, 0), -1)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img, (x, y), 3, (0, 0, 0), -1)


reset()
cv2.namedWindow("main_window")
cv2.setMouseCallback("main_window", writing)
while True:
    cv2.imshow("main_window", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        reset()
    elif key == ord('m'):
        mk_imgshow()
    elif key == ord('c'):
        ck_m()
    elif key == ord('q'):
        break
cv2.destroyAllWindows() 