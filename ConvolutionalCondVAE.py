import numpy as np 
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform, VarianceScaling

tf.compat.v1.disable_eager_execution()

class ConvCVAE (tf.keras.Model) :

    def __init__(self, 
                 label_dim,
                 latent_dim,
                 activation_fn = tf.nn.relu,
                 alpha = 1,
                 beta = 1,
                 image_dim = [64, 64, 3],               
                 learning_rate = 0.001,
                 batch_size = 32,
                 save_path = None,
                 load_model = None,
                 max_grad_norm = 10,
                 dropout = 0.5
                 ):

        """
        Parameters
        ----------
        label_dim : int 
            label vector dimension, 
        activation_fn 
                FC layers activation function [default tf.nn.relu]
        alpha : float
                alpha-beta VAE parameter for weighting the reconstruction loss [default 1]
        beta : float
                alpha-beta VAE parameter for weighting the KL divergence [default 1]
        learning_rate : float
                [default 0.001]
        batch_size : int
                batch size of the network [default 32]
        save_path : sring
                path of the model saver [default None] 
        load_model : string 
                path of the model loader [default None]
        max_grad_norm : float
                gradient clipping parameter [default 10]
        dropout : float
                dropout regularization parameter [deafult 0.5]

        """
        super(ConvCVAE, self).__init__()
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.activation_fn = activation_fn
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.dropout = dropout
        self.nn_type = "Convolutional"

        # Network generation
        self.build_graph()

        # Summary
        tf.compat.v1.summary.scalar("loss", self.loss)

        # Session
        self.sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(log_device_placement=False))

        # Summary writer for tensor board
        self.summary = tf.compat.v1.summary.merge_all()
        if save_path:
            self.writer = tf.compat.v1.summary.FileWriter(save_path, self.sess.graph)
        
        # Load model
        if load_model:
            tf.compat.v1.reset_default_graph()
            self.saver.restore(self.sess, load_model)
    
    def build_graph(self):
        """Create the Network and define the Loss function and the Optimizer"""

        # Conditional input
        self.x = tf.compat.v1.placeholder(tf.float32, shape = [None, self.image_dim[0], self.image_dim[1], self.image_dim[2]], name = "input")
        self.y = tf.compat.v1.placeholder(tf.float32, shape = [None, self.label_dim], name = "label")
        
        label = tf.reshape(self.y, [-1, 1, 1, self.label_dim]) #batch_size, 1, 1, label_size
        one = tf.ones([self.batch_size] + self.image_dim[0:-1] + [self.label_dim]) #batch_size, 64, 64, label_size
        label = one * label #batch_size, 64, 64, label_size
        conditional_input = tf.concat([self.x, label], axis=3) #batch_size, 64, 64, label_size + 3.
        n_channels = self.image_dim[2] + self.label_dim

        # Layers Initializer
        self.initializer = GlorotUniform()

        #----------Encoder Network-----------
        
        with tf.compat.v1.variable_scope("Encoder"):
            
            #------------Encoder structure------------
            self.encoder = tf.keras.Sequential(
              [
                tf.keras.layers.InputLayer(input_shape=self.image_dim[0:-1] + [n_channels]),
                
                tf.keras.layers.Conv2D( 
                    filters=8, 
                    kernel_size=3, 
                    strides=(2, 2), 
                    padding = 'same',
                    activation=None,
                    kernel_initializer=self.initializer),
                tf.keras.layers.Activation(self.activation_fn),

                tf.keras.layers.Conv2D( 
                    filters=16, 
                    kernel_size=3, 
                    strides=(2, 2), 
                    padding = 'same',
                    activation=None,
                    kernel_initializer=self.initializer),
                tf.keras.layers.Activation(self.activation_fn),
            
                tf.keras.layers.Conv2D( 
                    filters=32, 
                    kernel_size=3, 
                    strides=(2, 2), 
                    padding = 'same',
                    activation=None, 
                    kernel_initializer=self.initializer),
                tf.keras.layers.Activation(self.activation_fn),
               
                tf.keras.layers.Conv2D(
                    filters=64, 
                    kernel_size=3, 
                    strides=(2, 2), 
                    padding = 'same',
                    activation=None, 
                    kernel_initializer=self.initializer),
                tf.keras.layers.Activation(self.activation_fn),

                tf.keras.layers.Flatten(),
                
                # Output layer - No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim)
            ])

        # Mean and logaritmic variance of the latent space distribution
        self.z_mean, self.z_log_var = tf.split(self.encoder(conditional_input), num_or_size_splits=2, axis=1)

        #------------Reparametrization---------------
        eps = tf.random.normal(shape = (self.batch_size, self.latent_dim), mean = 0.0, stddev = 1.0)       
        self.z = tf.compat.v1.add(self.z_mean, tf.compat.v1.multiply(tf.math.exp(self.z_log_var * .5), eps))
        conditional_z = tf.compat.v1.concat([self.z, self.y], axis=1) # (batch_size, label_dim + latent_dim)
        
        #-----------Decoder Network-------------
        with tf.compat.v1.variable_scope("Decoder"):
            
            self.decoder = tf.keras.Sequential(
              [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim + self.label_dim,)),
                tf.keras.layers.Dense(units=4*4*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(4, 4, 32)),

                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='same',
                    activation=None,
                    kernel_initializer=self.initializer),
                tf.keras.layers.Activation(self.activation_fn),

                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='same',
                    activation=None,
                    kernel_initializer=self.initializer),
                tf.keras.layers.Activation(self.activation_fn),

                tf.keras.layers.Conv2DTranspose(
                    filters=16,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='same',
                    activation=None,
                    kernel_initializer=self.initializer),
                tf.keras.layers.Activation(self.activation_fn),

                tf.keras.layers.Conv2DTranspose(
                    filters=8,
                    kernel_size=3,
                    strides=(2, 2),
                    padding='same',
                    activation=None,
                    kernel_initializer=self.initializer),
                tf.keras.layers.Activation(self.activation_fn),

                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=3, 
                    kernel_size=3, 
                    strides=(1, 1), 
                    padding='same'),
              ]
            )
         
        # Output
        logits = self.decoder(conditional_z)
        self.generated_image = tf.nn.sigmoid(logits)

        #---------Loss Function-----------
        with tf.compat.v1.name_scope('Loss'):
            self.latent_loss = self.kl_divergence()
            self.reconstr_loss = self.mse()
            self.loss = self.latent_loss + self.reconstr_loss
            
            self.loss = tf.reduce_mean(self.loss)

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
       
        # Gradient clipping
        _var = tf.compat.v1.trainable_variables()
        grads, _ = tf.compat.v1.clip_by_global_norm(tf.compat.v1.gradients(self.loss, _var), self.max_grad_norm)
        self.train = optimizer.apply_gradients(zip(grads, _var))

            
        
        # Saver
        self.saver = tf.compat.v1.train.Saver()

    #------------------------------------
    # ELBO loss methods
    #---------------------------------

    def kl_divergence(self):
        """Computes the KL divergence KL(q(z | x) ∥ p(z | x))"""
        #z_log_var = tf.compat.v1.clip_by_value(self.z_log_var, clip_value_min=1e-10, clip_value_max=1e+2)
        kl = - 0.5 * tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=1)
        return  kl

    def mse(self):
        """Computes the reconstruction loss as MSE."""
        x = tf.keras.backend.batch_flatten(self.x)
        x_reconstr = tf.keras.backend.batch_flatten(self.generated_image)
        return tf.reduce_sum(tf.square(x - x_reconstr), axis = 1)

    #-------------------------------------------------
    # Reconstruction, Encoding and Decoding methods
    #-------------------------------------------------

    def reconstruct(self, inputs, label):
        """Reconstruct a given data. """
        #print(len(inputs), self.batch_size)
        assert len(inputs) == self.batch_size
        assert len(label) == self.batch_size
        return self.sess.run(self.generated_image, feed_dict={self.x: inputs, self.y: label})

    def encode(self, inputs, label):
        """ 
        Input -> latent space 
        """
        return self.sess.run(self.z_mean, feed_dict={self.x: inputs, self.y: label})

    def decode(self, label, z = None):
        """ 
        Generates data by sampling from the latent space.
        If z is None, z is drawn from prior in latent space.
        Otherwise, data for this point in latent space is generated.
        """
        if z is None:
            z = 0.0 + np.random.randn(self.batch_size, self.latent_dim) * 1.0
        return self.sess.run(self.generated_image, feed_dict={self.z: z, self.y: label})