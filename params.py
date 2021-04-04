import tensorflow as tf

dcgan_params = {
    'latent_dim': 100,
    'generator_conv_nonlinearity': tf.nn.leaky_relu,
    'generator_output_nonlinearity': tf.nn.sigmoid,
    'discriminator_conv_nonlinearity': tf.nn.leaky_relu,
    'discriminator_output_nonlinearity': tf.identity,

}