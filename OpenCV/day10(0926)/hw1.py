import sys
import typing
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget
import cv2
from PyQt5.QtWidgets import *
from keras.models import load_model
import numpy as np

class Mk_c(QMainWindow):
    def __init__(self):
        super().__init__
        self.setWindowTitle("세그먼트")
        self.setGeometry(200, 200, 420, 100)

        f_l_b = QPushButton("파일로드", self)
        mk_b = QPushButton("동작", self)
        f_s_b = QPushButton("파일저장", self)
        e_b = QPushButton("종료", self)
        self.label = QLabel(" ", self)

        f_l_b.setGeometry(10, 10, 100, 30)
        mk_b.setGeometry(110, 10, 100, 30)
        f_s_b.setGeometry(210, 10, 100, 30)
        e_b.setGeometry(310, 10, 100, 30)
        self.label.setGeometry(10, 50, 200, 30)

        f_l_b.clicked.connect(self.f_l_f)
        mk_b.clicked.connect(self.mk_f)
        f_s_b.clicked.connect(self.f_s_f)
        e_b.clicked.connect(self.e_f)

    def f_l_f(self):
        l_fname = QFileDialog.getOpenFileName(self, "파일로드", ',/')
        self.img = cv2.cvtColor(cv2.resize(cv2.imread(l_fname[0]), (160, 160)), cv2.COLOR_BGR2RGB)
        if self.img is None:
            sys.exit("Image load failed")    
        self.label.setText("Image load success")

    def mk_f(self):
        m = load_model("u_net.h5")
        self.py = m.predict(np.array([self.img]))[0]
        cv2.imshow("t", self.py)

    def f_s_f(self):
        s_fname = QFileDialog.getSaveFileName(self, "파일저장", './')
        cv2.imwrite(s_fname[0], self.py)

    def e_f(self):
        cv2.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
m_w = Mk_c()
m_w.show()
app.exec_()
