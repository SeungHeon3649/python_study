from PyQt5.QtWidgets import *
import sys
import cv2
import numpy as np


class Cut_bk_img(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("그림_제거")
        self.setGeometry(200, 200, 600, 100)

        f_l_b = QPushButton("파일 로드", self)
        b_i_b = QPushButton("붓그리기", self)
        p_b = QPushButton("+", self)
        m_b = QPushButton("-", self)
        c_b = QPushButton("잘라내기", self)
        f_s_b = QPushButton("파일 저장", self)
        e_b = QPushButton("종료", self)
        self.label = QLabel("프로그램이 켜집니다.", self)
        self.p_label = QLabel("", self)

        f_l_b.setGeometry(10, 10, 100, 30)
        b_i_b.setGeometry(110, 10, 100, 30)
        p_b.setGeometry(210, 10, 50, 30)
        m_b.setGeometry(260, 10, 50, 30)
        c_b.setGeometry(310, 10, 100, 30)
        f_s_b.setGeometry(410, 10, 100, 30)
        e_b.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 50, 200, 30)
        self.p_label.setGeometry(210, 50, 200, 30)


        f_l_b.clicked.connect(self.f_l_b_f)
        b_i_b.clicked.connect(self.b_i_b_f)
        p_b.clicked.connect(self.p_f)
        m_b.clicked.connect(self.m_f)
        c_b.clicked.connect(self.c_b_f)
        f_s_b.clicked.connect(self.f_s_b_f)
        e_b.clicked.connect(self.e_b_f)

        self.L_C, self.R_C = (0, 0, 255), (255, 0, 0)
        self.P_SIZE = 5

    def p_f(self):
        self.P_SIZE = min(30, self.P_SIZE + 1)
        self.p_label.setText(f"{self.P_SIZE}")
        
    def m_f(self):
        self.P_SIZE = max(1, self.P_SIZE - 1)
        self.p_label.setText(f"{self.P_SIZE}")

    def f_l_b_f(self):
        # 파일 로드
        l_fname = QFileDialog.getOpenFileName(self, "파일 로드", './')
        self.img = cv2.imread(l_fname[0])
        if self.img is None:
            print("Image load failed")
            sys.exit()
        self.label.setText("Image load success")
        self.show_img = self.img.copy()
        cv2.imshow("show_img", self.show_img)

        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.mask[:, :] = cv2.GC_PR_BGD

    def dw_b(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.show_img, (x, y), self.P_SIZE, self.L_C, -1)
            cv2.circle(self.mask, (x, y), 5, cv2.GC_FGD, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(self.show_img, (x, y), self.P_SIZE, self.R_C, -1)
            cv2.circle(self.mask, (x, y), 5, cv2.GC_BGD, -1)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(self.show_img, (x, y), self.P_SIZE, self.L_C, -1)
            cv2.circle(self.mask, (x, y), 5, cv2.GC_FGD, -1)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:
            cv2.circle(self.show_img, (x, y), self.P_SIZE, self.R_C, -1)
            cv2.circle(self.mask, (x, y), 5, cv2.GC_BGD, -1)

        cv2.imshow("show_img", self.show_img)
    
    def b_i_b_f(self):
        # 붓소환
        self.label.setText("붓 소환")
        self.p_label.setText(f"{self.P_SIZE}")
        cv2.setMouseCallback("show_img", self.dw_b)

    def c_b_f(self):
        # 잘라내기
        backgr = np.zeros((1, 65), np.float64)
        forgr = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.img, self.mask, None, backgr, forgr,
                    5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask == cv2.GC_BGD) | (self.mask == cv2.GC_PR_BGD), 0, 1).astype('uint8')
        self.cut_img = self.img * mask2[:, :, np.newaxis]
        cv2.imshow("cut_img", self.cut_img)

    def f_s_b_f(self):
        # 파일 저장
        s_fname = QFileDialog.getSaveFileName(self, "파일 로드", './', "JPEG Files (*.jpg);;PNG Files (*.png)")
        cv2.imwrite(s_fname[0], self.cut_img)

    def e_b_f(self):
        # 종료
       cv2.destroyAllWindows()
       self.close()

app = QApplication(sys.argv)
m_win = Cut_bk_img()
m_win.show()
app.exec_()

