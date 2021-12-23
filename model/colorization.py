
__author__ = 'Majd Jamal'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.models import load_model
from skimage.color import lab2rgb
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam

class Colorizer:

	def __init__(self, params):
		self.params = params


	def build_generator(self):
		Nnodes = 256 * 256 * 3
		model = Sequential()

		model.add(layers.Input(shape = (256, 256, 3)))
		model.add(layers.Flatten())

		model.add(layers.Dense(196608))
		model.add(layers.LeakyReLU(alpha=0.2))
		#model.add(layers.Reshape((256, 256, 3)))

		model.add(layers.Conv2DTranspose(128, (4,4), strides = (2,2), padding = 'same'))
		model.add(layers.LeakyReLU(alpha=0.2))

		model.add(layers.Conv2DTranspose(128, (4,4), strides = (2,2), padding = 'same'))
		model.add(layers.LeakyReLU(alpha=0.2))

		model.add(layers.Conv2D(3, (256,256), activation = 'tanh', padding = 'same'))

		model.summary()

		return model

	def build_discriminator(self):

		model = Sequential()
		model.add(layers.Input((256,256,3)))

		model.add(layers.Conv2D(128, (3,3), strides = (2,2), padding = 'same'))
		model.add(layers.LeakyReLU(alpha=0.2))
		#model.add(layers.BatchNormalization(momentum = 0.8))

		model.add(layers.Conv2D(128, (3,3), strides = (2,2), padding = 'same'))
		model.add(layers.LeakyReLU(alpha=0.2))
		#model.add(layers.BatchNormalization(momentum = 0.8))

		model.add(layers.Flatten())
		model.add(layers.Dropout(0.4))
		model.add(layers.Dense(1, activation = 'sigmoid'))

		opt = Adam(lr = 0.002, beta_1 = 0.5)

		model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])

		model.summary()

		return model


	def train(self, generator, discriminator, arch, data_grey, data_color):

		for epoch in tqdm(range(self.params.epochs)):

			for batch_id in tqdm(range(len(data_grey))):

				grey_batch = data_grey[batch_id]
				color_batch = data_color[batch_id]

				gen_colors = generator.predict(grey_batch)

				N_batch_pts = grey_batch.shape[0]

				disc_loss_real = discriminator.train_on_batch(color_batch, np.ones(N_batch_pts,))

				disc_loss_fake = discriminator.train_on_batch(gen_colors, np.zeros(N_batch_pts,))

				generator_loss = arch.train_on_batch(grey_batch, np.ones(N_batch_pts,))

				#print(f'Epoch {epoch}/{self.params.epochs}')

		return generator, discriminator, arch

	def main(self, data_grey, data_color):

		#"""

		#discriminator = self.build_discriminator()

		#discriminator.compile(
	    #		loss='binary_crossentropy',
	    #		optimizer = 'adam')


		generator = self.build_generator()
		"""
		generator.compile(
			loss='binary_crossentropy',
			optimizer = 'adam')


		z = layers.Input(shape = (256, 256, 3))

		colors = generator(z)

		discriminator.trainable = False

		judge = discriminator(colors)

		arch = Model(z, judge)

		arch.compile(loss = 'binary_crossentropy', optimizer = 'adam')

		generator, discriminator, arch = self.train(generator, discriminator, arch, data_grey, data_color)

		#"""
		#generator = load_model('generator.h5')
		#pred = generator.predict(data_grey[0][5].reshape((1, 256, 256, 3)))


		#plt.imshow(data_color[0][5])
		#plt.show()
		#print(pred)
		#plt.imshow(data_grey[0][5])
		#plt.show()


		#generator.save('generator.h5')
