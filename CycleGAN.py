from GAN import *
import numpy as np
import tensorflow.keras as keras

# ============================= CycleGAN =============================

class CycleGAN(Model):
    def __init__(self, monet_gen, monet_disc, photo_gen, photo_disc, lambda_cycle=10):
        super().__init__()
        self.monet_gen = monet_gen
        self.monet_disc = monet_disc
        self.photo_gen = photo_gen
        self.photo_disc = photo_disc
        self.lambda_cycle = lambda_cycle
    
    def get_monet_gen(self):
        return self.monet_gen
    def get_photo_gen(self):
        return self.photo_gen
    def get_photo_disc(self):
        return self.photo_disc
    def get_monet_disc(self):
        return self.monet_disc
    
    def compile(self, 
                monet_gen_opt, 
                monet_disc_opt,
                photo_gen_opt, 
                photo_disc_opt, 
                gen_loss, disc_loss, cycle_loss, id_loss=None):
        super().compile()
        self.monet_gen_opt = monet_gen_opt
        self.monet_disc_opt = monet_disc_opt
        self.photo_gen_opt = photo_gen_opt
        self.photo_disc_opt = photo_disc_opt
        
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.cycle_loss = cycle_loss
        self.id_loss = id_loss
    
    def train_step(self, batch):
        monet, photo = batch
        
        with tf.GradientTape(persistent=True) as tape:
            
            fake_monet = self.monet_gen(photo, training=True)
            cycled_photo = self.photo_gen(fake_monet, training=True)
            fake_photo = self.photo_gen(monet, training=True)
            cycled_monet = self.monet_gen(fake_photo, training=True)
            
            # ----
            same_monet = self.monet_gen(monet, training=True)
            same_photo = self.photo_gen(photo, training=True)
            
            # Disc outs
            disc_fake_monet = self.monet_disc(fake_monet)
            disc_fake_photo = self.photo_disc(fake_photo)
            disc_monet = self.monet_disc(monet)
            disc_photo = self.photo_disc(photo)
            
            # Cycle loss
            total_cylce_loss = self.cycle_loss(monet, cycled_monet, self.lambda_cycle) + self.cycle_loss(photo, cycled_photo, self.lambda_cycle)
            
            # Gen loss
            monet_gen_loss = self.gen_loss(disc_fake_monet)
            photo_gen_loss = self.gen_loss(disc_fake_photo)
            total_monet_gen_loss = monet_gen_loss + total_cylce_loss # + id???
            total_photo_gen_loss = photo_gen_loss + total_cylce_loss # + id???
            
            # Disc loss
            monet_disc_loss = self.disc_loss(disc_fake_monet, disc_monet)
            photo_disc_loss = self.disc_loss(disc_fake_photo, disc_photo)
            
            
        monet_gen_grad = tape.gradient(total_monet_gen_loss, self.monet_gen.trainable_variables)
        photo_gen_grad = tape.gradient(total_photo_gen_loss, self.photo_gen.trainable_variables)
        
        monet_disc_grad = tape.gradient(monet_disc_loss, self.monet_disc.trainable_variables)
        photo_disc_grad = tape.gradient(photo_disc_loss, self.photo_disc.trainable_variables)
        
        # Applying gradients
        
        self.monet_gen_opt.apply_gradients(zip(monet_gen_grad,
                                               self.monet_gen.trainable_variables))
        self.photo_gen_opt.apply_gradients(zip(photo_gen_grad,
                                               self.photo_gen.trainable_variables))
        self.monet_disc_opt.apply_gradients(zip(monet_disc_grad,
                                                self.monet_disc.trainable_variables))
        self.photo_disc_opt.apply_gradients(zip(photo_disc_grad,
                                                self.photo_disc.trainable_variables))
        
        return {
            'monet_gen_loss': total_monet_gen_loss,
            'photo_gen_loss': total_photo_gen_loss,
            'monet_disc_loss': monet_disc_loss,
            'photo_disc_loss': photo_disc_loss
        }
        
    def call(self, inputs, training=None, mask=None):
        images = self.monet_gen(inputs)
        return images

    def save_model(self, path):
        self.monet_gen.save(path + '/monet_gen')
        self.photo_gen.save(path + '/photo_gen')
        self.monet_disc.save(path + '/monet_disc')
        self.photo_disc.save(path + '/photo_disc')
        np.save(path + '/monet_gen_opt', self.monet_gen_opt.get_weights())
        np.save(path + '/photo_gen_opt', self.photo_gen_opt.get_weights())
        np.save(path + '/monet_disc_opt', self.monet_disc_opt.get_weights())
        np.save(path + '/photo_disc_opt', self.photo_disc_opt.get_weights())
        
# ============================= Loss =============================
with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1
    

# ============================= Model Initialization =============================

def load_model(path):
    monet_gen = models.load_model(path + '/monet_gen')
    photo_gen = models.load_model(path + '/photo_gen')
    monet_disc = models.load_model(path + '/monet_disc')
    photo_disc = models.load_model(path + '/photo_disc')
    
    gan = CycleGAN(monet_gen, monet_disc, photo_gen, photo_disc)
    
    monet_gen_opt = optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
    monet_disc_opt = optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
    photo_gen_opt = optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
    photo_disc_opt = optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
    
    monet_gen_vars = monet_gen.trainable_variables
    monet_gen_zero = [tf.zeros_like(w) for w in monet_gen_vars]
    monet_gen_opt.apply_gradients(zip(monet_gen_zero, monet_gen_vars))
    monet_gen_opt.set_weights(np.load(path + '/monet_gen_opt.npy', allow_pickle=True))
    
    photo_gen_vars = photo_gen.trainable_variables
    photo_gen_zero = [tf.zeros_like(w) for w in photo_gen_vars]
    photo_gen_opt.apply_gradients(zip(photo_gen_zero, photo_gen_vars))
    photo_gen_opt.set_weights(np.load(path + '/photo_gen_opt.npy', allow_pickle=True))
    
    monet_disc_vars = monet_disc.trainable_variables
    monet_disc_zero = [tf.zeros_like(w) for w in monet_disc_vars]
    monet_disc_opt.apply_gradients(zip(monet_disc_zero, monet_disc_vars))
    monet_disc_opt.set_weights(np.load(path + '/monet_disc_opt.npy', allow_pickle=True))
    
    photo_disc_vars = photo_disc.trainable_variables
    photo_disc_zero = [tf.zeros_like(w) for w in photo_disc_vars]
    photo_disc_opt.apply_gradients(zip(photo_disc_zero, photo_disc_vars))
    photo_disc_opt.set_weights(np.load(path + '/photo_disc_opt.npy', allow_pickle=True))
    
    
    gan.compile(monet_gen_opt=monet_gen_opt,
                monet_disc_opt=monet_disc_opt,
                photo_gen_opt=photo_gen_opt,
                photo_disc_opt=photo_disc_opt,
                gen_loss=gen_loss,
                disc_loss=disc_loss,
                cycle_loss=calc_cycle_loss)
    return gan
    

def uncompiled_CycleGAN():
    gan = CycleGAN(generator(), discriminator(), generator(), discriminator())
    return gan

def compiled_CycleGAN():
    gan = uncompiled_CycleGAN()
    gen_optimizer = optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
    disc_optimizer = optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999)
    gan.compile(monet_gen_opt=optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999),
                monet_disc_opt=optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999),
                photo_gen_opt=optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999),
                photo_disc_opt=optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.999),
                gen_loss=gen_loss,
                disc_loss=disc_loss,
                cycle_loss=calc_cycle_loss)
    return gan


