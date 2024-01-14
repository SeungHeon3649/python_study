from PyQt5.QtWidgets import *
import cv2
import numpy as np
import sys

class Panorama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("파노라마 영상")
        self.setGeometry(200,200,520,100)

        v_b= QPushButton("영상 수집", self)
        self.ck_ci_b = QPushButton("수집 영상 확인", self)
        self.ct_b = QPushButton("영상 연결", self)
        self.s_b = QPushButton("저장", self)
        e_b = QPushButton("종료", self)
        self.label = QLabel("프로그램이 켜집니다.", self)


        v_b.setGeometry(10,10,100,30)
        self.ck_ci_b.setGeometry(110,10,100,30)
        self.ct_b.setGeometry(210,10,100,30)
        self.s_b.setGeometry(310,10,100,30)
        e_b.setGeometry(410,10,100,30)
        self.label.setGeometry(10,10,520,80)

        #self.ck_ci_b.setEnabled(False)
        #self.ct_b.setEnabled(False)
        #self.s_b.setEnabled(False)
        v_b.clicked.connect(self.v_f)
        self.ck_ci_b.clicked.connect(self.ck_ci_b_f)
        self.ct_b.clicked.connect(self.ct_f)
        self.s_b.clicked.connect(self.s_f)
        e_b.clicked.connect(self.e_f)
        

    def v_f(self):
        #self.ck_ci_b.setEnabled(False)
        #self.ct_b.setEnabled(False)
        #self.s_b.setEnabled(False)
        self.label.setText("캡처는 c 로 진행 종료는 Esc")
        self.cam=cv2.VideoCapture(0)
        if not self.cam.isOpened(): sys.exit("카메라 연결 불가")

        self.imgs=[]
        while True:
            ret,img=self.cam.read()
            if not ret:
                print("캡처 불가")
                break
            cv2.imshow("v_img",img)
            key=cv2.waitKey(1)
            if key==ord('c'):
                self.imgs.append(img)
            elif key==27:
                self.cam.release()
                cv2.destroyAllWindows("v_img")
                break
            

        if len(self.imgs)>=2:
            self.ck_ci_b.setEnabled(True)
            self.ct_b.setEnabled(True)
            self.s_b.setEnabled(True)

    def ck_ci_b_f(self):
        self.label.setText(f"수집된 이미지의 장수는 {len(self.imgs)}")
        sk=cv2.resize(self.imgs[0],dsize=(0,0),fx=0.25,fy=0.25)
        for i in range(1,len(self.imgs)):
            sk=np.htack((sk,cv2.resize(self.imgs[i],dsize=(0,0),fx=0.25,fy=0.25)))
        cv2.imshow("ck_imgs",sk)

    def ct_f(self):
        stit = cv2.Stitcher_create()
        st,self.s_img = stit.stitch(self.imgs)
        if st == cv2.STITCHER_OK:
            cv2.imshow("end_img",self.s_img)
        else:
            self.label.setText("파노라마 제작에 실패했습니다.")

    def s_f(self):
        s_fname = QFileDialog.getSaveFileName(self,"파일저장","./")
        cv2.imwrite(s_fname[0],)

    def e_f(self):
        self.cam.release()
        cv2.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
m_win = Panorama()
m_win.show()
app.exec_()