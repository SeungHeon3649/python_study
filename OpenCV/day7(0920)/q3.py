# 동영상으로 비교

from PyQt5.QtWidgets import *
import sys
import cv2
import numpy as np


class Object_detecting(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("그림_제거")
        self.setGeometry(200, 200, 450, 100)

        f_l_b = QPushButton("검출대상 로드", self)
        v_on = QPushButton("찾기영상 생성", self)
        f_s_b = QPushButton("파일 저장", self)
        e_b = QPushButton("종료", self)
        self.label = QLabel("프로그램이 켜집니다.", self)

        f_l_b.setGeometry(10, 10, 100, 30)
        v_on.setGeometry(110, 10, 100, 30)
        f_s_b.setGeometry(210, 10, 100, 30)
        e_b.setGeometry(310, 10, 100, 30)
        self.label.setGeometry(10, 50, 200, 30)


        f_l_b.clicked.connect(self.f_l_b_f)
        v_on.clicked.connect(self.v_on_f)
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

    def v_on_f(self):
        self.label.setText("v_on_f 동작")
        cv2.destroyWindow("cut_img")
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened(): self.close()
        
        self.cut_img_gray = cv2.cvtColor(self.cut_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT.create()
        cut_kp, cut_desc = sift.detectAndCompute(self.cut_img_gray, None)

        while True:
            ret, self.frame = self.cam.read()
            if not ret: break
            self.frame = cv2.flip(self.frame, 1)
            self.frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            frame_kp, frame_desc = sift.detectAndCompute(self.frame_gray, None)
            if frame_desc is not None and len(frame_desc) > 1:
                flann_matcher = cv2.DescriptorMatcher.create(cv2.DescriptorMatcher_FLANNBASED)
                knn_matcher = flann_matcher.knnMatch(cut_desc, frame_desc, 2)
                good_matches = []
                for m, n in knn_matcher:
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
                if len(good_matches) > 10:
                    src_pts = np.float32([cut_kp[m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches])
                    
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                    h, w = self.cut_img.shape[:2]
                    box1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    box2 = cv2.perspectiveTransform(box1, M)
                    frame_dst = cv2.polylines(self.frame, [np.int32(box2)], True, 255, 3, cv2.LINE_AA)
                    self.final_dst = cv2.drawMatches(self.cut_img, cut_kp, frame_dst, frame_kp, good_matches, None,
                            flags = cv2.DrawMatchesFlags_DEFAULT)
                cv2.imshow("final_dst", self.final_dst)
            key = cv2.waitKey(10)
            if key == 27: 
                cv2.destroyWindow("video")
                break

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

