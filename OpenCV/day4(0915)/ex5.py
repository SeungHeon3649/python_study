import cv2
import skimage
import matplotlib.pyplot as plt
import numpy as np
img=skimage.data.coffee()
#cv2.imshow('t',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
slic_im=skimage.segmentation.slic(img,compactness=20,n_segments=600)
#print(img.shape,slic_im.shape)
sp_im=skimage.segmentation.mark_boundaries(img,slic_im)
sp_im1=np.uint8(sp_im*255)
cv2.imshow('t',cv2.cvtColor(sp_im1,cv2.COLOR_RGB2BGR))

slic_im=skimage.segmentation.slic(img,compactness=50,n_segments=300)
sp_im=skimage.segmentation.mark_boundaries(img,slic_im)
sp_im2=np.uint8(sp_im*255)
cv2.imshow('t1',cv2.cvtColor(sp_im2,cv2.COLOR_RGB2BGR))
#plt.imshow(slic_im)
#plt.show()



cv2.waitKey()