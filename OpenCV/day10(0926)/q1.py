# pyqt로 객체검출 기능 껐다 켰다

from PyQt5.QtWidgets import *
import cv2
from keras.models import load_model
import sys
import pickle
import numpy as np

class Yolov3(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("yolov3를 이용한 객체 검출기")
        self.setGeometry(200, 200, 450, 100)
        self.ck = False

        video = QPushButton("비디오 켜기", self)
        detection = QPushButton("객체 검출", self)
        pause = QPushButton("객체 검출중지", self)
        off = QPushButton("비디오 종료", self)
        self.label = QLabel("프로그램이 켜집니다.", self)

        video.setGeometry(10, 10, 100, 30)
        detection.setGeometry(110, 10, 100, 30)
        pause.setGeometry(210, 10, 100, 30)
        off.setGeometry(310, 10, 100, 30)
        self.label.setGeometry(10, 50, 400, 30)

        video.clicked.connect(self.video_on)
        detection.clicked.connect(self.object_detection)
        pause.clicked.connect(self.pause_f)
        off.clicked.connect(self.v_off_f)

    def video_on(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): self.close()
        
        while True:
            ret, self.frame = self.cap.read()
            if not ret: break
            self.frame2 = cv2.flip(self.frame, 1)

            if self.ck:
                model, out_ls, class_name = self.c_yolov3() # yolo모델 생성
                colors = np.random.uniform(0, 255, (len(class_name), 3)) #부류별 색상 결정
                res = self.yolov3_detect(self.frame, model, out_ls)
                for i  in range(len(res)):
                    sx, sy, ex, ey, conf, id = res[i]
                    text = str(class_name[id]) + f'{conf:.2f}'
                    cv2.rectangle(self.frame, (sx, sy), (ex, ey), colors[id], 2)
                    cv2.putText(self.frame, text, (sx, sy + 30), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                colors[id], 2)

            else:
                self.label.setText("비디오 동작중")
            
            cv2.imshow("frame", self.frame)
            key = cv2.waitKey(1)
            if key == 27: 
                cv2.destroyWindow("frame")
                break
    
    def c_yolov3(self):
        f = open('coco_names.txt', 'r')
        c_l = [i.strip() for i in f]
        m = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

        l_name = m.getLayerNames()
        out_ls = [l_name[i -1] for i in m.getUnconnectedOutLayers()]

        return m, out_ls, c_l

    def yolov3_detect(self, img, model, out_ls):
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
                if conf_d > 0.5: #신뢰도가 50퍼 이상인 정보 도출
                    c_x, c_y = int(vec85[0] * ow), int(vec85[1] * oh)
                    w, h = int(vec85[2] * ow), int(vec85[3] * oh)
                    x, y = int(c_x - w / 2), int(c_y - h / 2)
                    box.append([x, y, x + w, y + h])
                    conf.append(float(conf_d))
                    id.append(c_l_id)
        ind = cv2.dnn.NMSBoxes(box, conf, 0.5, 0.4)
        oj = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
        return oj

    def object_detection(self):
            self.label.setText("객체 검출 중")
            self.ck = True
            self.video_on()

    def pause_f(self):
        self.ck = False
        self.video_on()
    
    def v_off_f(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
m_win = Yolov3()
m_win.show()
app.exec_()