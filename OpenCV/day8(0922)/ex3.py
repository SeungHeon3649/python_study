import cv2
import numpy as np
import sys

def c_yolov3():
    f = open('coco_names.txt', 'r')
    c_l = [i.strip() for i in f]
    m = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    l_name = m.getLayerNames()
    out_ls = [l_name[i -1] for i in m.getUnconnectedOutLayers()]

    return m, out_ls, c_l

model, out_ls, class_name = c_yolov3() # yolo모델 생성
colors = np.random.uniform(0, 255, (len(class_name), 3)) #부류별 색상 결정

img = cv2.imread('soccer.jpg')
if img is None: sys.exit("Image load failed")

def yolov3_detect(img, model, out_ls):
    oh, ow = img.shape[0], img.shape[1]
    t_img = cv2.dnn.blobFromImage(img, 1.0 / 256, (416, 416), (0, 0, 0), swapRB = True)
    model.setInput(t_img)
    out_3 = model.forward(out_ls)

    box, conf, id = [], [], []
    for out_put in out_3:
        for vec85 in out_put:
            sc = vec85[5:]
            c_l_id = np.argmax(sc)
            conf_d = sc[c_l_id]
            if conf_d > 0.9: #신뢰도가 50퍼 이상인 정보 도출
                c_x, c_y = int(vec85[0] * ow), int(vec85[1] * oh)
                w, h = int(vec85[2] * ow), int(vec85[3] * oh)
                x, y = int(c_x - w / 2), int(c_y - h / 2)
                box.append([x, y, x + w, y + h])
                conf.append(float(conf_d))
                id.append(c_l_id)
    ind = cv2.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    oj = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
    return oj

# yolo 모델로 물체 검출
res = yolov3_detect(img, model, out_ls)

# 검출된 물체 영상 표시
for i  in range(len(res)):
    sx, sy, ex, ey, conf, id = res[i]
    text = str(class_name[id]) + f'{conf:.2f}'
    cv2.rectangle(img, (sx, sy), (ex, ey), colors[id], 2)
    cv2.putText(img, text, (sx, sy + 30), cv2.FONT_HERSHEY_PLAIN, 1.5,
                colors[id], 2)

cv2.imshow("oj_yolo_img", img)
cv2.waitKey()
cv2.destroyAllWindows()

