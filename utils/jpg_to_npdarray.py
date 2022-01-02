
__author__ = 'Majd Jamal'

import cv2
from skimage.color import rgb2lab
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def jpg_to_npdarray():
	""" Converts .jpg movie frames to pre-saved dataset model in np.ndarray format. 
	"""
	Npts = 7000
	dim = (256, 256)

	X = np.zeros((Npts, 256, 256, 3))
	Y = np.zeros((Npts, 256, 256, 3))


	for i in tqdm( range(Npts)):


		x = f'movie_frames/train/X/X{i}.jpg'
		y = f'movie_frames/train/Y/Y{i}.jpg'

		#read
		x = cv2.imread(x)
		x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
		x = cv2.resize(x, dim, interpolation = cv2.INTER_AREA)

		y = cv2.imread(y)
		y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
		y = cv2.resize(y, dim, interpolation = cv2.INTER_AREA)

		#store
		X[i] = x
		Y[i] = y


	np.save('processed_data/X.npy', X)
	np.save('processed_data/Y.npy', Y)
