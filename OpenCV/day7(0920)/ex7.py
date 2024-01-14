import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy

(train_x, train_y), (test_x, test_y) = mnist.load_data()
s_train_x = train_x.reshape(-1, 28, 28, 1) / 255.0
s_test_x = test_x.reshape(-1, 28, 28, 1) /255.0
s_train_y = to_categorical(train_y)
s_test_y = to_categorical(test_y)

m =Sequential()
m.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = s_train_x.shape[1:]))
m.add(Conv2D(32, (3, 3), activation = 'relu'))
m.add(MaxPooling2D(2))
m.add(Dropout(0.25))
m.add(Conv2D(64, (3, 3), activation = 'relu'))
m.add(Conv2D(64, (3, 3), activation = 'relu'))
m.add(MaxPooling2D(2))
m.add(Dropout(0.25))
m.add(Flatten())
m.add(Dense(10, activation = 'softmax'))
m.compile(optimizer = Adam(learning_rate = 0.001),
          loss = 'categorical_crossentropy', metrics = 'acc')
hy = m.fit(s_train_x, s_train_y, validation_data = (s_test_x, s_test_y),
           batch_size = 128, epochs = 100, verbose = 2)
m.save('m3.h5')

# 로드 모델 불러오기
# from keras.models import load_model

# # 저장된 모델을 불러옵니다.
# loaded_model = load_model('m3.h5')

# #예측
# predictions = loaded_model.predict(내 데이터)