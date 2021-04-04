import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

# Resnet
from models.dcgan.dcgan import Generator, Discriminator
from models.dcgan.script import DCGANTrainable

from utils.Logger import CheckpointLogger

def run(config):
    #1. Initialize loggers
    logger = CheckpointLogger(exp_name = config['exp_name'])

    #2. Data Loader
    if config['dataset'] == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        train_dataset = train_dataset.shuffle(buffer_size=100000).batch(batch_size=config['batch_size'])
        input_shape = train_images.shape[1:]
    elif config['dataset'] == 'mnist':
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        # Normalize the images to [-1, 1]
        train_images = train_images / 255.0
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(60000).batch(batch_size=config['batch_size'])
    else:
        raise ValueError(f"Please provide this dataset for {config['dataset']}")

    #3. Model Loader
    if config['exp_name'] == 'dcgan':
        generator = Generator(latent_dim=config['latent_dim'], output_height=28, output_width=28,
                              output_channels=1, conv_nonlinearity=tf.nn.leaky_relu,
                              output_nonlinearity=tf.nn.sigmoid, filters=32, kernel_size=3)

        discriminator = Discriminator(input_width=28, input_height=28, input_channels=1,
                                      filters=32, kernel_size=3, drop_rate=0.2, conv_nonlinearity=tf.nn.leaky_relu,
                                      output_nonlinearity=tf.identity)
    else:
        raise ValueError(f"Please provide this cnn_network for {config['exp_name']}")

    #4. Trainer
    if config['exp_name'] == 'dcgan':
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        dis_optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        trainer = DCGANTrainable(logger=logger, gen_optimizer=gen_optimizer, dis_optimizer=dis_optimizer,
                generator=generator, discriminator=discriminator, latent_dim=config['latent_dim'],
                train_dataset=train_dataset, test_dataset=None, epochs=config['epochs'],
                critic_step=1, log_frequency=1)
    else:
        raise ValueError(f"Please provide this cnn_network for {config['exp_name']}")


    #5. Write the config.json files
    logger.save_config(config)

    #6. Training
    trainer.run()


if __name__ == "__main__":
    # 1. Argument Parser
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, required=True) # The cnn_network you want to run
    parser.add_argument('--dataset', type=str, required=True) # The dataset you want to run
    args = parser.parse_args()

    params = {
        'exp_name': args.exp_name,
        'dataset': args.dataset,
        'epochs': 40,
        'batch_size': 256,
        'learning_rate': 1e-4,
        'latent_dim': 100,
    }

    run(params)