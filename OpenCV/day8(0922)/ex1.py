# 압축풀기
#tar -xvf annotation.tar
#tar -xvf images.tar
#tar -xvf lists.tar

# import os
# import shutil
# import pathlib
# from keras.utils import image_dataset_from_directory
# from keras.applications.densenet import DenseNet121
# from keras.layers import Flatten, Dense, Rescaling, Dropout
# from keras.models import Sequential
# from keras.losses import sparse_categorical_crossentropy
# from keras.optimizers import Adam

# d_path = pathlib.Path('Images')

# tr_data = image_dataset_from_directory(d_path, batch_size = 32,
#                                           image_size = (224, 224))

# m = DenseNet121(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
# n_m = Sequential()
# n_m.add(Rescaling(1.0 / 255.0))
# n_m.add(m)
# n_m.add(Flatten())
# n_m.add(Dense(1024, activation = "relu"))
# n_m.add(Dropout(0.75))
# n_m.add(Dense(120, activation = 'softmax'))
# n_m.compile(optimizer = Adam(learning_rate = 0.000001), loss = 'sparse_categorical_crossentropy', metrics = 'acc')
# hy = n_m.fit(tr_data, epochs =10, verbose = 2)

from PyQt5.QtWidgets import *
import cv2
from keras.models import load_model
import sys
import pickle
import numpy as np

m = load_model("cnn_for_stanford_dogs.h5")
dog_name = pickle.load(open('dog_species_names.txt', 'rb'))

class Dog_s_g(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("강아지 종 분류기")
        self.setGeometry(200, 200, 600, 100)

        im_in_b = QPushButton("강아지 사진 입력", self)
        ck_b = QPushButton("강아지 품종 확인:" , self)
        e_b = QPushButton("종료", self)

        im_in_b.setGeometry(10, 10, 150, 30)
        ck_b.setGeometry(160, 10, 150, 30)
        e_b.setGeometry(310, 10, 50, 30)

        im_in_b.clicked.connect(self.im_in_f)
        ck_b.clicked.connect(self.ck_f)
        e_b.clicked.connect(self.e_f)

    def im_in_f(self):
        fname = QFileDialog.getOpenFileName(self, "강아지 로드", './')
        self.img = cv2.imread(fname[0])
        if self.img is None: sys.exit("파일 없음")
        
        cv2.imshow("dog_img", self.img)

    def ck_f(self):
        x = np.reshape(cv2.resize(self.img, (224, 224)), (1, 224, 224, 3))
        out_d = m.predict(x)[0]
        t_5 = np.argsort(-out_d)[:5]
        t_5_dog_name = [dog_name[i] for i in t_5]
        for i in range(5):
            p ='(' + str(out_d[t_5[i]]) + ')'
            name = str(t_5_dog_name[i]).split('-')[1]
            cv2.putText(self.img, p + name, (10, 100 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("dog_img", self.img)
    
    def e_f(self):
        cv2.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
m_win = Dog_s_g()
m_win.show()
app.exec_()