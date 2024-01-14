from PyQt5.QtWidgets import *
import mediapipe as mp
import sys
import cv2

class Cut_bk_img(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mediapipe")
        self.setGeometry(200, 200, 450, 100)

        video = QPushButton("비디오 켜기", self)
        f_l_b = QPushButton("파일 로드", self)
        e_b = QPushButton("종료", self)
        self.label = QLabel("프로그램이 켜집니다.", self)

        video.setGeometry(10, 10, 100, 30)
        f_l_b.setGeometry(110, 10, 100, 30)
        e_b.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 50, 200, 30)

        video.clicked.connect(self.video_on)
        f_l_b.clicked.connect(self.f_l_b_f)
        e_b.clicked.connect(self.e_b_f)


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

    def f_l_b_f(self):
        # 파일 로드
        l_fname = QFileDialog.getOpenFileName(self, "파일 로드", './')
        print(l_fname)
        self.img = cv2.imread(l_fname[0], cv2.IMREAD_UNCHANGED)
        if self.img is None:
            print("Image load failed")
            sys.exit()
        self.label.setText("Image load success")
        cv2.imshow("self_img", self.img)

    def dw_b(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ptx, self.pty = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dst = self.img[self.pty:y, self.ptx:x]
            cv2.imshow("cut_img", self.dst)
    
    def c_b_f(self):
        # 잘라내기
        self.label.setText("잘라내기")
        cv2.setMouseCallback("self_img", self.dw_b)

    def f_s_b_f(self):
        # 파일 저장
        s_fname = QFileDialog.getSaveFileName(self, "파일 저장", './', "JPEG Files (*.jpg);;PNG Files (*.png)")
        print(s_fname)
        cv2.imwrite(s_fname[0], self.dst)

    def e_b_f(self):
        # 종료
       cv2.destroyAllWindows()
       self.close()

app = QApplication(sys.argv)
m_win = Cut_bk_img()
m_win.show()
app.exec_()



in_img = cv2.imread('love.png', cv2.IMREAD_UNCHANGED)
in_img = cv2.resize(in_img, (0, 0), fx = 0.1, fy = 0.1)
in_h, in_w = in_img.shape[:2]

# 일종의 패키지를 불러온 것
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5)

cam = cv2.VideoCapture(0)

while True:
    r, f = cam.read()

    rec = face_detection.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    if rec.detections:
        for det in rec.detections:
            point_dot = mp_face_detection.get_key_point(det, mp_face_detection.FaceKeyPoint.LEFT_EYE)
            sx, sy = int(point_dot.x * f.shape[1] - in_w // 2), int(point_dot.y * f.shape[0] - in_h // 2) 
            ex, ey = int(point_dot.x * f.shape[1] + in_w // 2), int(point_dot.y * f.shape[0] + in_h // 2) 
            
            if sx > 0 and sy > 0 and ex < f.shape[1] and ey < f.shape[0]:
                # 4채널중 마지막이 png이미지의 알파값, 투명도
                ap = in_img[:, :, 3:] / 255
                f[sy : ey, sx : ex] = f[sy : ey, sx : ex] * (1 - ap) + in_img[:, :, :3] * ap
            
    cv2.imshow('w', f)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()