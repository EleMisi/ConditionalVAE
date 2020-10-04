import numpy as np 
import tensorflow as tf
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose


#########################
#        ENCODER        #
#########################

class Encoder(tf.keras.Model):

  def __init__(self, latent_dim):
    
    super(Encoder, self).__init__()

    self.enc_block_1 = Conv2D( 
                        filters=32, 
                        kernel_size=3, 
                        strides=(2, 2), 
                        padding = 'same',
                        kernel_initializer=he_normal())
    
    
    self.enc_block_2 = Conv2D( 
                  filters=64, 
                  kernel_size=3, 
                  strides=(2, 2), 
                  padding = 'same',
                  kernel_initializer=he_normal())
    
    
    
    self.enc_block_3 = Conv2D( 
                  filters=128, 
                  kernel_size=3, 
                  strides=(2, 2), 
                  padding = 'same',
                  kernel_initializer=he_normal())
            

    
    self.enc_block_4 = Conv2D( 
                  filters=256, 
                  kernel_size=3, 
                  strides=(2, 2), 
                  padding = 'same',
                  kernel_initializer=he_normal())
    
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(latent_dim + latent_dim)  


  def __call__(self, conditional_input, latent_dim, is_train):
     # Encoder block 1
    x = self.enc_block_1(conditional_input)
    x = BatchNormalization(trainable = is_train)(x)
    x = tf.nn.leaky_relu(x)
    # Encoder block 2
    x = self.enc_block_2(x)
    x = BatchNormalization(trainable = is_train)(x)
    x = tf.nn.leaky_relu(x)
    # Encoder block 3
    x = self.enc_block_3(x)
    x = BatchNormalization(trainable = is_train)(x)
    x = tf.nn.leaky_relu(x)
    # Encoder block 4
    x = self.enc_block_4(x)
    x = BatchNormalization(trainable = is_train)(x)
    x = tf.nn.leaky_relu(x)   

    x = self.dense(self.flatten(x))

    return x



#########################
#        DECODER        #
#########################

class Decoder(tf.keras.Model):
    

  def __init__(self):

    super(Decoder, self).__init__()

    self.dense = tf.keras.layers.Dense(4*4*32*8)
    self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 32*8))

    self.dec_block_1 = Conv2DTranspose(
            filters=256,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

    self.dec_block_2 = Conv2DTranspose(
            filters=128,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())
        
    self.dec_block_3 = Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

    self.dec_block_4 = Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            kernel_initializer=he_normal())

    self.dec_block_5 = Conv2DTranspose(
            filters=3, 
            kernel_size=3, 
            strides=(1, 1), 
            padding='same',
            kernel_initializer=he_normal())

  def __call__(self, z_cond, is_train):
    # Reshape input
    x = self.dense(z_cond)
    x = tf.nn.leaky_relu(x)
    x = self.reshape(x)
    # Decoder block 1
    x = self.dec_block_1(x)
    x = BatchNormalization(trainable = is_train)(x)
    x = tf.nn.leaky_relu(x)
    # Decoder block 2
    x = self.dec_block_2(x)
    x = BatchNormalization(trainable = is_train)(x)
    x = tf.nn.leaky_relu(x)
    # Decoder block 3
    x = self.dec_block_3(x)
    x = BatchNormalization(trainable = is_train)(x)
    x = tf.nn.leaky_relu(x)
    # Decoder block 4
    x = self.dec_block_4(x)
    x = BatchNormalization(trainable = is_train)(x)
    x = tf.nn.leaky_relu(x)

    return self.dec_block_5(x)



#########################
#       Conv-CVAE       #
#########################

class ConvCVAE (tf.keras.Model) :

    def __init__(self, 
        encoder,
        decoder,
        label_dim,
        latent_dim,
        beta = 1,
        image_dim = [64, 64, 3]):

        super(ConvCVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.beta = beta = 1
        self.image_dim = image_dim = [64, 64, 3]              


    def __call__(self, inputs, is_train):
    
        input_img, input_label, conditional_input = self.conditional_input(inputs, batch_size = 32)

        z_mean, z_log_var = tf.split(self.encoder(conditional_input, self.latent_dim, is_train), num_or_size_splits=2, axis=1)    
        z_cond = self.reparametrization(z_mean, z_log_var, input_label)
        logits = self.decoder(z_cond, is_train)

        recon_img = tf.nn.sigmoid(logits)

        # Loss computation #
        latent_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1) # KL divergence
        reconstr_loss = np.prod((64,64)) * tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(input_img), tf.keras.backend.flatten(recon_img)) # over weighted MSE
        loss = reconstr_loss + self.beta * latent_loss # weighted ELBO loss
        loss = tf.reduce_mean(loss) 

        return {
                  'recon_img': recon_img,
                  'latent_loss': latent_loss,
                  'reconstr_loss': reconstr_loss,
                  'loss': loss
              }


    def conditional_input(self, inputs, batch_size):
        """ Builds the conditional input and returns the original input images, their labels and the conditional input."""

        input_img = tf.keras.layers.InputLayer(input_shape=self.image_dim, dtype = 'float32')(inputs[0])
        input_label = tf.keras.layers.InputLayer(input_shape=(self.label_dim,), dtype = 'float32')(inputs[1])
        labels = tf.reshape(inputs[1], [-1, 1, 1, self.label_dim]) #batch_size, 1, 1, label_size
        ones = tf.ones([batch_size] + self.image_dim[0:-1] + [self.label_dim]) #batch_size, 64, 64, label_size
        labels = ones * labels #batch_size, 64, 64, label_size
        conditional_input = tf.keras.layers.InputLayer(input_shape=(self.image_dim[0], self.image_dim[1], self.image_dim[2] + self.label_dim), dtype = 'float32')(tf.concat([inputs[0], labels], axis=3))

        return input_img, input_label, conditional_input


    def reparametrization(self, z_mean, z_log_var, input_label):
        """ Performs the riparametrization trick"""

        eps = tf.random.normal(shape = (32, self.latent_dim), mean = 0.0, stddev = 1.0)       
        z = z_mean + tf.math.exp(z_log_var * .5) * eps
        z_cond = tf.concat([z, input_label], axis=1) # (batch_size, label_dim + latent_dim)

        return z_cond
