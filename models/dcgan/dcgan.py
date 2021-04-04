import tensorflow as tf
from types import FunctionType

class Generator(tf.keras.Model):
    def __init__(self, latent_dim: int, output_height: int, output_width: int,
                 output_channels: int, conv_nonlinearity: FunctionType, output_nonlinearity: FunctionType,
                 filters: int, kernel_size: int):
        '''

        :param latent_dim: noise dimension z
        :param output_height: output image height, e.g. 28 for MNIST
        :param output_width: output image height, e.g. 28 for MNIST
        :param output_channels: output image height, e.g. 28 for MNIST
        :param conv_nonlinearity:
        :param output_nonlinearity:
        :param filters:
        :param kernel_size:
        '''
        super(Generator, self).__init__()

        # Initial size is 4 times smaller than image size, so we need 2 times upsampling
        init_size_height = output_height // 4
        init_size_width = output_width // 4
        init_size_channel = 256

        # Step 1. Make input layer
        inputs = tf.keras.Input(shape=(latent_dim,))
        # Step 2. Convert latent to conv2d
        X = tf.keras.layers.Dense(units = init_size_height * init_size_width * init_size_channel)(inputs)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation(conv_nonlinearity)(X)

        X = tf.keras.layers.Reshape((init_size_height, init_size_width, init_size_channel))(X)
        # Step 3. Upsampling 2 times and reducing filter size
        X = tf.keras.layers.Conv2DTranspose(filters=filters/2, kernel_size=kernel_size,
                                            strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation(conv_nonlinearity)(X)

        # Step 4. Convert channels to same output channels, no upsampling more
        X = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=kernel_size,
                                            strides=2, padding='same')(X)
        X = tf.keras.layers.Activation(output_nonlinearity)(X)

        self.model = tf.keras.Model(inputs = inputs, outputs=X)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)


class Discriminator(tf.keras.Model):
    def __init__(self, input_width: int, input_height: int, input_channels: int,
                 filters: int, kernel_size: int, drop_rate: float, conv_nonlinearity: FunctionType,
                 output_nonlinearity: FunctionType):
        super(Discriminator, self).__init__()

        inputs = tf.keras.Input(shape = (input_height, input_width, input_channels))

        X = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same')(inputs)
        X = tf.keras.layers.Activation(conv_nonlinearity)(X)
        X = tf.keras.layers.Dropout(rate=drop_rate)(X)

        X = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same')(X)
        X = tf.keras.layers.Activation(conv_nonlinearity)(X)
        X = tf.keras.layers.Dropout(rate=drop_rate)(X)

        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(units=1)(X)
        X = tf.keras.layers.Activation(output_nonlinearity)(X)

        self.model = tf.keras.Model(inputs = inputs, outputs = X)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)





