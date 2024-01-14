import os
import shutil
import pathlib
from keras.utils import image_dataset_from_directory
from keras.applications.densenet import DenseNet121
from keras.layers import Flatten, Dense, Rescaling, Dropout
from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

d_path = pathlib.path('train')
def f(n, st_idx, end_idx):
    for dog in ("dog", ):
        dir = d_path /n/dog
        os.makedirs(dir)
        f_ns = [f'{dog}.{i}.jpg' for i in range(st_idx, end_idx)]
        for f_n in f_ns:
            shutil.copyfile(src = d_path/f_n, dst = dir/f_n)

f('train_data', 0, 1000)
l_path = pathlib.path('trian/tr_data')
data1 = image_dataset_from_directory(l_path, batch_size = 32, image_siez = (224, 244))


m =DenseNet121(include_top = False, weight = 'imagenet', input_shape = (224, 224, 3))
n_m = Sequential()
n_m.add(Rescaling(1.0 / 255.0))
n_m.add(m)
n_m.add(Flatten())
n_m.add(Dense(1024, activation = "relu"))
n_m.add(Dropout(0.75))
n_m.add(Dense(120, activation = 'softmax'))
n_m.compile(optimizer = Adam(learning_rate = 0.000001), loss = 'sparse_categorical_crossentropy', metrics = 'acc')
hy = n_m.fit()


