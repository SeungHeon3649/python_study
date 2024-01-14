import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

m = ResNet50(weights = 'imagenet')

img = cv2.imread("rabbit.jpg")
x = np.reshape(cv2.resize(img, (224, 224)), (1, 224, 224, 3))
s_x = preprocess_input(x)
py = m.predict(s_x)
t_5 = decode_predictions(py, top = 5)[0]
for i in range(5):
    cv2.putText(img, t_5[i][1] + ':' + str(t_5[i][2]), (10, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("ck_img", img)
cv2.waitKey()
cv2.destroyAllWindows()
