import numpy as np
import cv2
import sys

def cs_yolo_v3():
    f = open('coco_names.txt', 'r')
    class_nm = [i.strip() for i in f.readline()]

    m = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    l_name = m.getLayerNames()
    out_l = [l_name[i -1] for i in m.getUnconnectedOutLayers()]

    return m, out_l, class_nm

def yolo_sc(img, m, out_l):
    oh, ow = img.shape[0], img.shape[1]
    tt_img = cv2.dnn.blobFromImage(img, 1.0 / 256, (448, 448),
                                    (0, 0, 0), swapRB = True)
    m.setInput(tt_img)
    out_p = m.forward(out_l)

    box, conf, id = [], [], [] # 박스, 신뢰도, 부류번호
    for out in out_p:
        for vec85 in out:
            sc = vec85[5:]
            class_id = np.argmax(sc)
            confn = sc[class_id]
            if confn > 0.5: #신뢰도가 50퍼 이상인 정보 도출
                c_x, c_y = int(vec85[0] * ow), int(vec85[1] * oh)
                w, h = int(vec85[2] * ow), int(vec85[3] * oh)
                x, y = int(c_x - w / 2), int(c_y - h / 2)
                box.append([x, y, x + w, y + h])
                conf.append(float(confn))
                id.append(class_id)
    idx = cv2.dnn.NMSBoxes(box, conf, 0.5, 0.4)
    oj = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in idx]
    return oj

m, out_l, class_nm = cs_yolo_v3()
colors = np.random.uniform(0, 255, (100, 3))

from sort import Sort
sort = Sort()

cap = cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit("카메라 인식 실패")

while True:
    ret, frame = cap.read()
    if not ret: sys.exit("동작불가")
    red = yolo_sc(frame, m, out_l)
    persons = [red[i] for i in range(len(red)) if red[i][5] == 0]

    if len(persons) == 0:
        tk = sort.update()
    else:
        tk = sort.update(np.array(persons))
    
    for i in range(len(tk)):
        x1, y1, x2, y2, tk_idx = tk[i].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[tk_idx], 2)
        cv2.putText(frame, str(tk_idx), (x1 + 10, y1 + 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, colors[tk_idx], 2)
        
    cv2.imshow("tracking_ps_SORT", frame)
    
    key = cv2.waitKey(1)
    if key == 27: break

cap.release()
cv2.destroyAllWindows()
