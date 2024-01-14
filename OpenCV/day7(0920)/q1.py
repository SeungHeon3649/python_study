# 1. 사진 영역 cut 이미지 생성기 만들기(파일로드, cut, 저장)
# 2. ex2.py 동작으로 배경제거
# 3. 대상 검출기 만들기(검출대상 로드, 검출정보 로드, 찾기)
# 번외 영상으로 접근
# pyQt를 이용하여 프로그램을 완성하시오

from PyQt5.QtWidgets import *
import sys
import cv2
import numpy as np


class Cut_bk_img(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("그림_제거")
        self.setGeometry(200, 200, 450, 100)

        f_l_b = QPushButton("파일 로드", self)
        c_b = QPushButton("잘라내기", self)
        f_s_b = QPushButton("파일 저장", self)
        e_b = QPushButton("종료", self)
        self.label = QLabel("프로그램이 켜집니다.", self)

        f_l_b.setGeometry(10, 10, 100, 30)
        c_b.setGeometry(110, 10, 100, 30)
        f_s_b.setGeometry(210, 10, 100, 30)
        e_b.setGeometry(310, 10, 100, 30)
        self.label.setGeometry(10, 50, 200, 30)

        f_l_b.clicked.connect(self.f_l_b_f)
        c_b.clicked.connect(self.c_b_f)
        f_s_b.clicked.connect(self.f_s_b_f)
        e_b.clicked.connect(self.e_b_f)

    def p_f(self):
        self.P_SIZE = min(30, self.P_SIZE + 1)
        self.p_label.setText(f"{self.P_SIZE}")
        
    def m_f(self):
        self.P_SIZE = max(1, self.P_SIZE - 1)
        self.p_label.setText(f"{self.P_SIZE}")

    def f_l_b_f(self):
        # 파일 로드
        l_fname = QFileDialog.getOpenFileName(self, "파일 로드", './')
        print(l_fname)
        self.img = cv2.imread(l_fname[0])
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

