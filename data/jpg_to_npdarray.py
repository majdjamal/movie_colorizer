
__author__ = 'Majd Jamal'

import cv2
from skimage.color import rgb2lab
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Npts = 5421
dim = (256, 256)

X_train = np.zeros((Npts, 256, 256, 3))
Y_train = np.zeros((Npts, 256, 256, 3))



for i in tqdm( range(Npts)):

	#source
	x = f'movie_frames/train/X/X{i}.jpg'
	y = f'movie_frames/train/Y/Y{i}.jpg'

	#read
	x = cv2.imread(x)
	y = cv2.imread(y)

	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)

	#re-size
	x = cv2.resize(x, dim, interpolation = cv2.INTER_AREA)
	y = cv2.resize(y, dim, interpolation = cv2.INTER_AREA)


	#convert color space
	#lab_X = rgb2lab(x)
	#lab_Y = rgb2lab(y)

	#store
	X_train[i] = x
	Y_train[i] = y

#save
np.save('X.npy', X_train)
np.save('Y.npy', Y_train)
