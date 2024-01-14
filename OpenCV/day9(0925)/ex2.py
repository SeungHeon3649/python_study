from keras.utils import Sequence
from keras.models import Model
from keras.layers import Input,Activation,Conv2D,BatchNormalization,SeparableConv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,add
import os
from keras.preprocessing.image import load_img
import numpy as np
import random

input_dir='./images/'
target_dir='./annotations/trimaps/'
img_size=(160,160)
n_calss=3 #분할 레이블(1:물체,2:배경,3:경계)
batch_size=32

img_paths=sorted([os.path.join(input_dir,f)
                  for f in os.listdir(input_dir)
                  if f.endswith('.jpg')])
label_paths=sorted([os.path.join(target_dir,f)
                    for f in os.listdir(target_dir)
                    if f.endswith('.png') and not f.startswith('.')])

class N_m_data(Sequence):
    def __init__(self,batch_size,img_size,img_paths,label_paths):
        self.batch_size=batch_size
        self.img_size=img_size
        self.img_paths=img_paths
        self.label_paths=label_paths

    def __len__(self):
        return len(self.label_paths)//self.batch_size

    def __getitem__(self, index):
        idx=index*self.batch_size
        batch_img_paths=self.img_paths[idx:idx+self.batch_size]
        batch_label_paths=self.label_paths[idx:idx+self.batch_size]
        x=np.zeros((self.batch_size,)+self.img_size+(3,),'float32')
        for i,path in enumerate(batch_img_paths):
            img = load_img(path,target_size=self.img_size)
            x[i] = img
        y=np.zeros((self.batch_size,)+self.img_size+(1,),'uint8')
        for i,path in enumerate(batch_label_paths):
            img = load_img(path,target_size=self.img_size,color_mode='grayscale')
            y[i] = np.expand_dims(img,2)
            y[i]-=1 #부류번호 1,2,3 -> 0,1,2
        return x,y

random.Random(1).shuffle(img_paths)
random.Random(1).shuffle(label_paths)
tt_samp=int(len(img_paths)*0.1)#10%태스트 데이터
tr_img_paths=img_paths[:-tt_samp]
tr_label_paths=label_paths[:-tt_samp]
tt_img_paths=img_paths[-tt_samp:]
tt_label_paths=label_paths[-tt_samp:]

tr_dataset=N_m_data(batch_size,img_size,tr_img_paths,tr_label_paths)
tt_dataset=N_m_data(batch_size,img_size,tt_img_paths,tt_label_paths)

from keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

m = load_model('u_net_m.h5')

t_img, t_mask = next(iter(tt_dataset))

py = m.predict(t_img)
pre_masks = tf.math.argmax(py, axis = -1)
pre_masks = pre_masks[..., tf.newaxis]
py = pre_masks[0]
oj_img = t_img.astype('uint8')[0]
ty = t_mask[0]
plt.subplot(1, 3, 1)
plt.imshow(py)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(ty)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(oj_img)
plt.axis('off')
plt.show()

import cv2
py = m.predict(tt_dataset)
cv2.imshow("t", cv2.imread(tt_img_paths[0]))
cv2.imshow("oj", cv2.imread(tt_label_paths[0]) * 64)
cv2.imshow("pr", cv2.imread(py[0]))
cv2.waitKey()
cv2.destroyAllwindows()

