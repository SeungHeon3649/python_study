# 사진으로 비교

from PyQt5.QtWidgets import *
import sys
import cv2
import numpy as np


class Object_detecting(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("그림_제거")
        self.setGeometry(200, 200, 550, 100)

        f_l_b = QPushButton("검출대상 로드", self)
        f_l_b2 = QPushButton("검출정보 로드", self)
        find_pt = QPushButton("찾기이미지 생성", self)
        f_s_b = QPushButton("파일 저장", self)
        e_b = QPushButton("종료", self)
        self.label = QLabel("프로그램이 켜집니다.", self)

        f_l_b.setGeometry(10, 10, 100, 30)
        f_l_b2.setGeometry(110, 10, 100, 30)
        find_pt.setGeometry(210, 10, 100, 30)
        f_s_b.setGeometry(310, 10, 100, 30)
        e_b.setGeometry(410, 10, 100, 30)
        self.label.setGeometry(10, 50, 200, 30)


        f_l_b.clicked.connect(self.f_l_b_f)
        f_l_b2.clicked.connect(self.f_l_b2_f)
        find_pt.clicked.connect(self.find_pt)
        f_s_b.clicked.connect(self.f_s_b_f)
        e_b.clicked.connect(self.e_b_f)

    def f_l_b_f(self):
        # 검출대상 로드
        l_fname = QFileDialog.getOpenFileName(self, "검출대상 로드", './')
        self.cut_img = cv2.imread(l_fname[0])
        if self.cut_img is None:
            print("Image load failed")
            sys.exit()
        self.label.setText("Image load success")
        cv2.imshow("cut_img", self.cut_img)

    def f_l_b2_f(self):
        # 검출정보 로드
        l_fname = QFileDialog.getOpenFileName(self, "검출정보 로드", './')
        self.img = cv2.imread(l_fname[0])
        if self.img is None:
            print("Image load failed")
            sys.exit()
        self.label.setText("Image load success")
        cv2.imshow("img", self.img)

    def find_pt(self):
        #찾기 이미지 생성
        self.cut_img_gray = cv2.cvtColor(self.cut_img, cv2.COLOR_BGR2GRAY)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT.create()
        cut_kp, cut_desc = sift.detectAndCompute(self.cut_img_gray, None)
        img_kp, img_desc = sift.detectAndCompute(self.img_gray, None)

        flann_matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matcher = flann_matcher.knnMatch(cut_desc, img_desc, 2)
        good_matches = []
        for m, n in knn_matcher:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        if len(good_matches) > 10:
            src_pts = np.float32([cut_kp[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([img_kp[m.trainIdx].pt for m in good_matches])

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            h, w = self.cut_img.shape[:2]
            box1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            box2 = cv2.perspectiveTransform(box1, M)
            self.dst = cv2.polylines(self.img, [np.int32(box2)], True, 255, 3, cv2.LINE_AA)
            self.final_dst = cv2.drawMatches(self.cut_img, cut_kp, self.dst, img_kp, good_matches, None,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        cv2.imshow("final_dst", self.final_dst)

    def f_s_b_f(self):
        # 파일 저장
        s_fname = QFileDialog.getSaveFileName(self, "파일 로드", './', "JPEG Files (*.jpg);;PNG Files (*.png)")
        cv2.imwrite(s_fname[0], self.final_dsdt)

    def e_b_f(self):
        # 종료
        cv2.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
m_win = Object_detecting()
m_win.show()
app.exec_()

