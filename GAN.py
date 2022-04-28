import os, random, json, PIL, shutil, re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa

from tensorflow.keras import Model, losses, optimizers, models
import Images
from Parameters import *

strategy = tf.distribute.get_strategy()


# ============================= Auxillary functions =============================
"""
    downsample(n, sz, in, st)
    outputs Conv2D s.t.
        # filters = n
        dim of kernel = sz
        normal initialization
        instance_normalization if in
        strides = st
        padding with 0s
        without bias
        and LeakyReLU as activation
"""
def downsample(filters, size, apply_instancenorm=True, strides=2):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, 
                                                    stddev=0.02)
    
    result = tf.keras.Sequential()
    result.add(L.Conv2D(filters, 
                        size, 
                        strides=strides, 
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False))
    
    # batch is of size 1
    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(
            gamma_initializer=gamma_init))
    
    result.add(L.LeakyReLU())
    return result

"""
    upsample(n, sz, ad, st)
    outputs DeConv2d s.t.
        # filters = n
        dim of kernel = sz
        normal initialization
        always uses IN
        strides = st
        padding with 0s
        without bias
        Dropout(0.5) if ad
        and ReLU
"""
def upsample(filters, size, apply_dropout=False, strides=2):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, 
                                                    stddev=0.02)
    result = tf.keras.Sequential()
    #Adding deconvolution
    result.add(L.Conv2DTranspose(filters, size, 
                                strides=strides,
                                padding='same',
                                kernel_initializer=initializer,
                                use_bias=False))
    # Always adding IN
    result.add(tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(L.Dropout(0.5))
    
    result.add(L.ReLU())
    return result

# ============================= Generator =============================

def generator():
    inputs = L.Input(shape=[HEIGHT, WIDTH, CHANNELS])

    down_stack = [
        downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
        downsample(128, 4),                          # (bs, 64, 64, 128)
        downsample(256, 4),                          # (bs, 32, 32, 256)
        downsample(512, 4),                          # (bs, 16, 16, 512)
        downsample(512, 4),                          # (bs, 8, 8, 512)
        downsample(512, 4),                          # (bs, 4, 4, 512)
        downsample(512, 4),                          # (bs, 2, 2, 512)
        downsample(512, 4),                          # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4),                     # (bs, 16, 16, 1024)
        upsample(256, 4),                     # (bs, 32, 32, 512)
        upsample(128, 4),                     # (bs, 64, 64, 256)
        upsample(64, 4),                      # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = L.Conv2DTranspose(CHANNELS, 4,
                             strides=2,
                             padding='same',
                             kernel_initializer=initializer,
                             activation='tanh') # (bs, 256, 256, 3)

    x = inputs
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = L.Concatenate()([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)

# ============================= Discriminator =============================

def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = L.Input(shape=[HEIGHT, WIDTH, CHANNELS], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = L.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = L.Conv2D(512, 4, strides=1,
                    kernel_initializer=initializer,
                    use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = L.LeakyReLU()(norm1)

    zero_pad2 = L.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = L.Conv2D(1, 4, strides=1,
                    kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return Model(inputs=inp, outputs=last)

# ============================= GAN =============================

class GAN(Model):
    def __init__(self, gen, disc, lambda_cycle=10):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.lambda_cycle = lambda_cycle
    
    def get_gen(self):
        return self.gen
    def get_disc(self):
        return self.disc
    
    def compile(self, gen_optimizer, disc_optimizer, gen_loss, disc_loss, id_loss):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.id_loss = id_loss
    
    def call(self, inputs, training=None, mask=None):
        images = self.gen(inputs)
        pred = self.disc(images)
        return images, pred
    
    def train_step(self, batch):
        monet, photo = batch
        
        # Losses
        with tf.GradientTape(persistent=True) as tape:
            fake_monet = self.gen(photo, training=True)
            
            disc_fake_monet = self.disc(fake_monet, 
                                        training=True)
            disc_real_monet = self.disc(monet, 
                                        training=True)
            
            disc_loss = self.disc_loss(disc_fake_monet, disc_real_monet)
            gen_loss = self.gen_loss(disc_fake_monet)
            total_gen_loss = gen_loss #+ self.id_loss(photo, fake_monet)
            
        
        # Gradients 
        gen_gradients = tape.gradient(total_gen_loss,
                                        self.gen.trainable_variables)
        disc_gradients = tape.gradient(disc_loss,
                                        self.disc.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.gen.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )
        return {
            'disc_loss' : disc_loss,
            'gen_loss'  : total_gen_loss
        }
        
# ============================= Loss functions =============================

with strategy.scope():
    def binCross():
            return losses.BinaryCrossentropy(
                from_logits=True,
                reduction=losses.Reduction.NONE
            )
        
    def disc_loss(fake, real):
        # {0 -> fake, 1 -> real}
        real_loss = binCross()(tf.ones_like(real), real)
        fake_loss = binCross()(tf.zeros_like(fake), fake)
        return (real_loss + fake_loss) * 0.5
    
    def gen_loss(disc_out):
        return binCross()(tf.ones_like(disc_out), disc_out)
    
    def id_loss(photo, monet, LAMBDA=0.001):
        loss = tf.reduce_mean(tf.abs(photo - monet))
        return LAMBDA * 0.5 * loss
    
# =============================
def uncompiled_GAN():
    monet_gen = generator()
    monet_disc = discriminator()
        
    gan = GAN(monet_gen, monet_disc)
    #gan.build(input_shape=[HEIGHT, WIDTH, CHANNELS])
    return gan

def compiled_GAN():
    gan = uncompiled_GAN()
    gen_optimizer = optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
    disc_optimizer = optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
    gan.compile(gen_optimizer,
                disc_optimizer,
                gen_loss=gen_loss,
                disc_loss=disc_loss,
                id_loss=id_loss)
    return gan
    