import os, random, json, PIL, shutil, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import tensorflow.keras as keras
from tensorflow.keras import Model, losses, optimizers, models
import Images
import GAN
import CycleGAN
import sys
from Parameters import *

# ============================= TPU =============================

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
print(tf.__version__)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# ============================= Datasets =============================

MONET_FILENAMES = ['monet_tfrec/' + f for f in os.listdir('monet_tfrec/')]
PHOTO_FILENAMES = ['photo_tfrec/' + f for f in os.listdir('photo_tfrec/')]

monet_ds = Images.load_dataset(MONET_FILENAMES, AUTOTUNE).batch(1)
photo_ds = Images.load_dataset(PHOTO_FILENAMES, AUTOTUNE).batch(1)

train_ds = Images.get_gan_dataset(MONET_FILENAMES, 
                                  PHOTO_FILENAMES, 
                                  AUTOTUNE,
                                  batch_size=BATCH_SIZE)

# ============================= Initialization =============================


def make_or_restore_model():
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return CycleGAN.load_model(latest_checkpoint)
    print("Creating a new model")
    return CycleGAN.compiled_CycleGAN()

class CycleGANcallback(keras.callbacks.Callback):
    def __init__(self):
        it = iter(photo_ds)
        for i in range(20):
            next(it)
        self.image_sample = next(it)
    
    def on_epoch_end(self, epoch, logs = None):
        
        gen_sample = self.model.predict(self.image_sample)
        
        plt.subplot(121)
        plt.title("input image")
        plt.imshow(self.image_sample[0] * 0.5 + 0.5)
        plt.axis('off')
        
        plt.subplot(122)
        plt.title("Generated image")
        plt.imshow(gen_sample[0] * 0.5 + 0.5)
        plt.axis('off')
        plt.savefig('Samples/sample_' + str(epoch + 1))
        
        if epoch % 5 == 4:
            print("\n========Saving model========\n")
            self.model.save_model(checkpoint_dir + "/cyclegan_" + str(epoch))

callbacks = [
        CycleGANcallback()
    ]

# ============================= Training =============================

with strategy.scope():
    
    gan = make_or_restore_model()
    
    it = iter(train_ds)
    gan(next(it)[0])
    
    if len(sys.argv) > 1:
        EPOCHS = int(sys.argv[1])
    
    history = gan.fit(train_ds, 
                steps_per_epoch=(n_monet//BATCH_SIZE),
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1).history
    
    gan.save_model('models/final_CycleGAN')
