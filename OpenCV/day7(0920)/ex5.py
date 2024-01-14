import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy

(train_x, train_y), (test_x, test_y) = mnist.load_data()
s_train_x = train_x.reshape(-1, 28, 28, 1) / 255.0
s_test_x = test_x.reshape(-1, 28, 28, 1) /255.0
s_train_y = to_categorical(train_y)
s_test_y = to_categorical(test_y)

m2 =Sequential()
m2.add(Conv2D(6, (5, 5), padding = 'same', activation = 'relu', input_shape = s_train_x.shape[1:]))
m2.add(MaxPooling2D((2, 2), 2))
m2.add(Conv2D(16, (5, 5), padding = 'valid', activation = 'relu'))
m2.add(MaxPooling2D((2, 2), 2))
m2.add(Conv2D(120, (5, 5), padding = 'valid', activation = 'relu'))
m2.add(Flatten())
m2.add(Dense(83, activation = 'relu'))
m2.add(Dense(10, activation = 'softmax'))
m2.compile(optimizer = Adam(learning_rate = 0.001),
          loss = 'categorical_crossentropy', metrics = 'acc')
hy2 = m2.fit(s_train_x, s_train_y, validation_data = (s_test_x, s_test_y),
           batch_size = 128, epochs = 30, verbose = 2)
m2.save('m2.h5')