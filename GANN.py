import tensorflow as tf
from tensorflow import keras
from loguru import logger
from tensorflow.keras.models import Model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tqdm as tqdm
from tensorflow.python.framework.ops import disable_eager_execution
gpus = tf.config.list_logical_devices('GPU')
mirrored_strategy = tf.distribute.MirroredStrategy(devices=gpus,cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


BATCH_SIZE = 3
with mirrored_strategy.scope():
    training_dir = "autoencoder/pics/bryan1"
    training_dir_uch = "autoencoder/pics/uchtdorf1"
    image_size = (256, 256)
    

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            # preprocessing_function=add_noise
            )
    validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            # preprocessing_function=add_noise
            )
    

    train_generator = train_datagen.flow_from_directory(
            training_dir,
            target_size = image_size,
            subset="training",
            batch_size=BATCH_SIZE,
            class_mode=None,
            color_mode="grayscale",
            seed=42,shuffle=True)
    validation_generator = validation_datagen.flow_from_directory(
            training_dir,
            target_size=image_size,
            batch_size=BATCH_SIZE,
            class_mode=None,
            color_mode="grayscale",
            subset="validation",
            seed=42)



    train_generator_uch = train_datagen.flow_from_directory(
            training_dir_uch,
            target_size = image_size,
            subset="training",
            batch_size=BATCH_SIZE,
            class_mode=None,
            color_mode="grayscale",
            seed=42,shuffle=True)
    validation_generator_uch = validation_datagen.flow_from_directory(
            training_dir_uch,
            target_size=image_size,
            batch_size=BATCH_SIZE,
            class_mode=None,
            color_mode="grayscale",
            subset="validation",
            seed=42)
    
     
    latent_space_dim = 32
    input_shape=(256,256,1)
    '''encoder'''
    encoder_input = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", strides=1)(encoder_input)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=2)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", strides=4)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", strides=1)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    shape_before_flatten = K.int_shape(net)[1:]
    net = tf.keras.layers.Flatten()(net)

    # custom layer - will not run on DPU
    encoder_z = tf.keras.layers.Dense(latent_space_dim)(net)

    # encoder_mu,encoder_log_variance outputs go to loss function
    # encoder_z is encoded latent space
    encoder=Model(inputs=encoder_input, outputs=[encoder_z])


    ''' decoder '''
    decoder_input = tf.keras.layers.Input(shape=latent_space_dim)
    net = tf.keras.layers.Dense(units=np.prod(shape_before_flatten))(decoder_input)
    net = tf.keras.layers.Reshape(target_shape=shape_before_flatten)(net)
    net = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=1)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=4)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding="same", strides=2)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.LeakyReLU()(net)
    decoder_output = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), padding="same", activation="sigmoid",strides=1)(net)


    # standard sigmoid
    #decoder_output = Activation('sigmoid')(net)

    decoder = Model(inputs=decoder_input, outputs=decoder_output)
    image_dim = 256
    image_chan = 3
    input_layer = tf.keras.layers.Input(shape=(image_dim,image_dim,image_chan))
    encoder_z = encoder.call(input_layer)

    dec_out = decoder.call(encoder_z)
    G = Model(inputs=input_layer, outputs=dec_out)
    
with mirrored_strategy.scope():
    def discriminator():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[256, 256, 1]))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dense(1))
        return model
    
    D = discriminator()
    G = G

    
print(G.summary())
print(D.summary())

with mirrored_strategy.scope():
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True,reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss



    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    

    checkpoint_dir = 'autoencoder/checkpoints/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     G=G,
                                     D=D)
    

    EPOCHS = 50
    num_examples_to_generate = BATCH_SIZE

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate,256,256,1])
    
    

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, 256,256,1])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = G(noise, training=True)

            real_output = D(images, training=True)
            fake_output = D(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, G.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, D.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, G.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, D.trainable_variables))
    @tf.function
    def generate_and_save_images(model, epoch, test_input):
      # Notice `training` is set to False.
      # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    @tf.function
    def train(dataset, epochs):
        for epoch in range(epochs):
            print("starting epoch", epoch)
            start = time.time()
            n = 0
            for image_batch in dataset:
                print(f"    starting step {n} of {len(dataset)}")
                train_step(image_batch)
                n += 1
                if n > len(dataset):
                    break
                

            # Produce images for the GIF as you go
            # display.clear_output(wait=True)
            print("got to generate image")
            generate_and_save_images(G,
                                     epoch + 1,
                                     seed)
            print("done generating image")
        # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

            # Generate after the final epoch
            # display.clear_output(wait=True)
            generate_and_save_images(G,epochs,seed)
train(train_generator_uch, EPOCHS)
encoder.save('gann_encoder.h5')
decoder.save('gann_decoder.h5')
G.save("generator.h5")
D.save("discriminator.h5")