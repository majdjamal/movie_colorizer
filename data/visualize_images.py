
__author__ = 'Majd Jamal'

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from skimage.color import lab2rgb

X = np.load('processed_data/X_val.npy')
y = np.load('processed_data/y_val.npy')

#x_test = lab2rgb(X[4])
#y_test = lab2rgb(y[4])

#plt.imshow(X[4])
#plt.show()

#plt.imshow(y_test)
#plt.show()