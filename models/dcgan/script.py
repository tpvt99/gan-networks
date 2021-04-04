import tensorflow as tf
import math
from utils.Logger import CheckpointLogger
import matplotlib.pyplot as plt

class DCGANTrainable():
    def __init__(self, logger: CheckpointLogger, gen_optimizer, dis_optimizer, generator, discriminator, latent_dim: int,
                 train_dataset, test_dataset, epochs: int, critic_step: int, log_frequency: int):
        self.logger = logger
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.critic_step = critic_step
        self.log_frequency = log_frequency

        # For logging metrics
        self.train_loss_gen = tf.keras.metrics.Mean()
        self.train_loss_dis = tf.keras.metrics.Mean()

        # Initializing checkpoints
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                         discriminator_optimizer=dis_optimizer,
                                         generator=generator,
                                         discriminator=discriminator,
                                         step=tf.Variable(0))
        self.logger.setup_tf_checkpoint(checkpoint=self.checkpoint, max_to_keep=5)


        # For generating images
        number_of_images_to_generate = 16
        self.seed = tf.random.normal([number_of_images_to_generate, self.latent_dim])

    def train_discriminator_step(self, data, labels):
        with tf.GradientTape() as tape:
            z = tf.random.normal(shape=(data.shape[0], self.latent_dim))
            generated_images = self.generator(z, training=True)
            generated_outputs = self.discriminator(generated_images, training=True)
            real_outputs = self.discriminator(data, training=True)

            # minimize -log(D(x)) + -log(1-D(G(z))
            loss1 = -tf.math.log(tf.nn.sigmoid(real_outputs))  # -log(D(x))
            loss2 = -tf.math.log(1 - tf.nn.sigmoid(generated_outputs))  # -log(1 - D(G(z))
            loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2)
        gradients = tape.gradient(loss, self.discriminator.trainable_variables)
        self.dis_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return loss

    def train_generator_step(self, data, labels):
        with tf.GradientTape() as tape:
            z = tf.random.normal(shape=(data.shape[0], self.latent_dim))
            generated_images = self.generator(z, training=True)
            generated_outputs = self.discriminator(generated_images, training=True)

            # minimize -log(D(G(x))
            loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(generated_outputs)))
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return loss

    def generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).

        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            #plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.imshow(predictions[i, :, :, 0] * 255.0, cmap='gray')
            plt.axis('off')

        self.logger.save_images(plt, 'image_at_epoch_{:04d}.png'.format(epoch))


    @tf.function
    def train_step(self, data, labels):
        for _ in range(self.critic_step):
            dis_loss = self.train_discriminator_step(data, labels)
        gen_loss = self.train_generator_step(data, labels)

        self.train_loss_gen.update_state(gen_loss)
        self.train_loss_dis.update_state(dis_loss)


    def run(self):
        step = int(self.checkpoint.step) # step = epoch if no resuming, else step is bigger than epoch
        total_runs = step + self.epochs
        for epoch in range(self.epochs):
            self.train_loss_gen.reset_states()
            self.train_loss_dis.reset_states()

            for index, (data, labels) in enumerate(self.train_dataset):
                self.train_step(data, labels)
                print(f'Epoch {step}/{total_runs}: Train {index}/{len(self.train_dataset)} | '
                  f'Train Loss Gen: {self.train_loss_gen.result():.5f} | '
                  f'Train Loss Dis: {self.train_loss_dis.result():.5f}')

            self.generate_and_save_images(self.generator, int(self.checkpoint.step), self.seed)

            if int(self.checkpoint.step) % self.log_frequency == 0:
                self.logger.save_checkpoint()

            step = int(self.checkpoint.step.assign_add(1))

            """ ------------------- Logging Stuff --------------------------"""
            self.logger.log_tabular('Epoch', step)
            self.logger.log_tabular('Train-Gen-Loss', self.train_loss_gen.result())
            self.logger.log_tabular('Train-Dis-Loss', self.train_loss_dis.result())
            self.logger.dump_tabular()

            self.logger.tf_board_save_scaler(step, self.train_loss_gen.result(), 'loss-gen', 'train')
            self.logger.tf_board_save_scaler(step, self.train_loss_dis.result(), 'loss-dis', 'train')

