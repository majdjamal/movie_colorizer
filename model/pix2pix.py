
__author__ = 'Majd Jamal'

import cv2
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from utils.utils import get_image
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow import random_normal_initializer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model

class Pix2Pix:

    def __init__(self, params, train_dataset):
        self.params = params
        self.train_dataset = train_dataset

        self.generator = None
        self.discriminator = None

        self.eta = params.eta
        self.beta_1 = params.beta_1

        self.loss_object = BinaryCrossentropy(from_logits = True)
        self.GeneratorOptimizer = Adam(2e-4, beta_1 = 0.5)
        self.DiscriminatorOptimizer = Adam(2e-4, beta_1 = 0.5)

    def DownBlock(self, filters: int, size: int, apply_batchnorm: bool = True) -> Sequential:
        """ Downscaling block. Used to build Pix2Pix network arcitecture.
        :params filters: number of output channels
        :params size: kernel size
        :params apply_batchnorm: To apply batch normalization
        :return block: A downscaling block.
        """
        initializer = random_normal_initializer(0., 0.02)

        block = Sequential()

        block.add(
            layers.Conv2D(filters, size,
                strides = 2,
                padding = "same",
                kernel_initializer = initializer,
                use_bias = False)
        )

        if apply_batchnorm:
            block.add(layers.BatchNormalization())

        block.add(layers.LeakyReLU())

        return block

    def UpscaleBlock(self, filters: int, size: int, apply_dropout = False):
        """ Upscaling block. Used to build Pix2Pix network arcitecture.
        :params filters: number of output channels
        :params size: kernel size
        :params apply_dropout: To apply dropout regularization.
        :return block: An upscaling block.
        """
        initializer = random_normal_initializer(0., 0.02)

        block = Sequential()

        block.add(
            layers.Conv2DTranspose(filters, size,
            strides = 2,
            padding = 'same',
            kernel_initializer = initializer,
            use_bias = False
            ))

        block.add(
            layers.BatchNormalization()
            )

        if apply_dropout:
            block.add(layers.Dropout(0.5))

        block.add(layers.ReLU())

        return block

    def MakeGenerator(self) -> Model:
        """ Builds the generator

        :return: Non-compiled generator
        """

        input = layers.Input((256,256,3))

        down_stack = [

            self.DownBlock(64, 4, apply_barchnorm = False),
            self.DownBlock(128, 4),
            self.DownBlock(256, 4),
            self.DownBlock(512, 4),
            self.DownBlock(512, 4),
            self.DownBlock(512, 4),
            self.DownBlock(512, 4),
            self.DownBlock(512, 4),
        ]

        up_stack = [
            self.UpscaleBlock(512, 4, apply_dropout = True),
            self.UpscaleBlock(512, 4, apply_dropout = True),
            self.UpscaleBlock(512, 4, apply_dropout = True),
            self.UpscaleBlock(512, 4),
            self.UpscaleBlock(256, 4),
            self.UpscaleBlock(128, 4),
            self.UpscaleBlock(64, 4),
        ]

        initializer = random_normal_initializer(0., 0.02)

        final = layers.Conv2DTranspose(3, 4,
            strides = 2,
            padding = 'same',
            kernel_initializer = initializer,
            activation = 'tanh')

        x = input

        skips = []

        for block in down_stack:
            x = block(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):

            x = up(x)
            x = layers.Concatenate()([x, skip])

        x = final(x)

        return Model(inputs = input, outputs = x)

    def generator_loss(self, disc_out, gen_out, target):
        """ Loss function for the generator.

        :params disc_out:
        :params gen_out:
        :params target:

        :return total_gen_loss: total generator loss
        :return gan_loss: adversarial loss
        :return l1_loss: l1 loss
        """

        gan_loss = self.loss_object(tf.ones_like(disc_out), disc_out)

        l1_loss = tf.reduce_mean(tf.abs(target - gen_out))

        total_gen_loss = gan_loss + l1_loss * 100

        return total_gen_loss, gan_loss, l1_loss

    def MakeDiscriminator(self) -> Model:
        """ Builds the discriminator

        :return: Non-compiled discriminator
        """


        initializer = random_normal_initializer(0.,0.02)

        inp = layers.Input(shape = [256, 256, 3])
        tar = layers.Input(shape = [256, 256, 3])

        x = layers.concatenate([inp, tar])

        down1 = self.DownBlock(64,4, False)(x)
        down2 = self.DownBlock(128, 4)(down1)
        down3 = self.DownBlock(256, 4)(down2)

        zero_pad1 = layers.ZeroPadding2D()(down3)

        conv = layers.Conv2D(512, 4,
            strides = 1,
            kernel_initializer = initializer,
            use_bias = False)(zero_pad1)

        batchnorm = layers.BatchNormalization()(conv)

        leaky_relu = layers.LeakyReLU()(batchnorm)

        zero_pad2 = layers.ZeroPadding2D()(leaky_relu)

        last = layers.Conv2D(1,4,
            strides = 1,
            kernel_initializer = initializer)(zero_pad2)

        return Model(inputs = [inp, tar], outputs = last)

    def discriminator_loss(self, disc_real_output, disc_gen_out):
        """ Loss function for the discriminator.

        :params disc_real_output: Discriminator loss on true image
        :params disc_gen_out: Discriminator loss on generated image

        :return total_disc_loss: Total discriminator loss
        """

        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

        gen_loss = self.loss_object(tf.ones_like(disc_gen_out), disc_gen_out)

        total_disc_loss = real_loss + gen_loss

        return total_disc_loss

    def generate_images(self, model: Model, test_input, tar, step: int) -> None :
        """ Generates some samples to monitor the learning progress.
        Samples are saved as .jpg in data/result folder.

        :params model: Generator
        :params test_input: Grayscale image from the training set
        :params tar: True colors for the test_input image
        :params step: Current iteration in the training process
        """

        pred = model(test_input, training = True)
        plt.figure(figsize=(15,15))

        display_list = [test_input[0], tar[0], pred[0]]

        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.imshow(display_list[i])
            plt.axis('off')

        plt.savefig(f'data/result/progress_step_{step}')

    @tf.function
    def train_step(self, input_image, target, step):
        """ Updated weights for one training step
        :params input_image: Grayscale image from the training set
        :params target: True colors for the input image
        :params step: Current iteration in the training process
        """

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)

        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)


        self.GeneratorOptimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )

        self.DiscriminatorOptimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )


    def fit(self, train_ds, steps: int) -> None:
        """ Start the network training process.
        :params train_ds: Training dataset
        :params steps: Number of training iterations.
        """

        start = time.time()

        for step, (input_image, target) in tqdm(train_ds.repeat().take(steps).enumerate()):

            if (step) % 1000 == 0:
                # Visualize progress

                display.clear_output(wait=True)
                self.generate_images(self.generator, input_image, target, step)

            if (steps) % 5000 == 0 and steps != 0:
                # Save weights after some steps

                self.generator.save('data/weights/movie_colorizer_train.h5')


            self.train_step(input_image, target, step)

    def predict(self, path: str = None) -> None:
        """ Predict colors for a black and white movie frame and saves
        the prediction as a jpg file.

        :params path: Path to image, e.g. 'data/movie_frames/X5.jpg'
        """

        try:
            self.generator = load_model('data/weights/movie_converter.h5')
        except:
            raise ValueError('Weights does not exist! Train your network and try again! Or send a message to majdj@kth.se to obtain a pre-trained model. ')

        x = get_image(path)

        pred = self.generator(x, training = False)

        pred = pred[0]
        pred = np.array(pred)
        pred = pred[...,::-1].copy()

        plt.imshow(pred)
        plt.axis('off')
        plt.savefig(f'data/result/frame_prediction/colorized_frame.jpg')
        plt.close()

    def main(self) -> None:

        self.generator = self.MakeGenerator()
        self.discriminator = self.MakeDiscriminator()

        if self.params.train:

            ds = self.train_dataset
            steps = self.params.steps

            self.fit(ds, steps)

        elif self.params.predict:
            test_path = self.params.test_path
            self.predict(test_path)

        else:
            raise ValueError('Select an operation. E.g. train with --train')
