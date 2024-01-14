from PyQt5.QtWidgets import *
import sys
import cv2

class Button(QMainWindow):
    def __init__(self):
        super().__init__()
        self.bt = QPushButton("버튼", self)
        self.bt.move(20, 20)
    def f(self):
        print("버튼 동작")

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("비디오")
        self.setGeometry(200, 200, 500, 100)

        v_on = QPushButton("비디오 킴", self)
        v_cc = QPushButton("비디오 캡쳐", self)
        v_sv = QPushButton("비디오 저장", self)
        v_off = QPushButton("비디오 종료", self)
        self.label = QLabel("프로그램이 켜집니다.", self)

        v_on.setGeometry(10, 10, 100, 30)
        v_cc.setGeometry(110, 10, 100, 30)
        v_sv.setGeometry(210, 10, 100, 30)
        v_off.setGeometry(310, 10, 100, 30)
        self.label.setGeometry(10, 50, 400, 30)

        v_on.clicked.connect(self.v_on_f)
        v_cc.clicked.connect(self.v_cc_f)
        v_sv.clicked.connect(self.v_sv_f)
        v_off.clicked.connect(self.v_off_f) 

    def v_on_f(self):
        self.label.setText("v_on_f 동작")
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened(): self.close()
        while True:
            ret, self.frame = self.cam.read()
            if not ret: break
            self.frame = cv2.flip(self.frame, 1)
            cv2.imshow("video", self.frame)
            key = cv2.waitKey(10)
            if key == 27: 
                cv2.destroyWindow("video")
                break

    def v_cc_f(self):
        self.label.setText("v_cc_f 동작")
        self.cap_frame = self.frame
        cv2.imshow("cap_img", self.cap_frame)

    def v_sv_f(self):
        self.label.setText("v_sv_f 동작")
        fname = QFileDialog.getSaveFileName(self, "파일 저장", './')
        cv2.imwrite(fname[0], self.cap_frame)

    def v_off_f(self):
        self.label.setText("v_off_f 동작")
        self.cam.release()
        cv2.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
m_win = Video()
m_win.show()
app.exec_()

