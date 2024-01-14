import keras.datasets as ds
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pl

(train_x, train_y), (test_x, test_y) = ds.cifar10.load_data()
s_train_x = train_x / 255.0
s_train_x = s_train_x[:15,]
s_train_y = train_y[:15,]


im_g = ImageDataGenerator(rotation_range = 20.0, width_shift_range = 0.2,
                           height_shift_range = 0.2, horizontal_flip = True)
gen_img = im_g.flow(s_train_x, s_train_y, batch_size = 4)
print(s_train_x.shape)
print(gen_img)