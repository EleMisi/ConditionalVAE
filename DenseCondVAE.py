import numpy as np 
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform, VarianceScaling

tf.compat.v1.disable_eager_execution()


class DenseCVAE (tf.keras.Model) :

    def __init__(self, 
                 label_dim,
                 nn_architecture,
                 activation_fn = tf.nn.relu,
                 alpha = 1,
                 beta = 1,
                 image_dim = 64*64*3,
                 latent_dim = 32,
                 learning_rate = 0.001,
                 batch_size = 32,
                 save_path = None,
                 load_model = None,
                 max_grad_norm = 1,
                 dropout = 0.5,
                 is_train = True
                 ):

        """
        Parameters
        -------------

        label_dim : int 
            label vector dimension, 
        nn_architecture : dict 
                "hidden_enc_j_dim": dimensionality of the j-th hidden layer output space (encoder)
                "hidden_dec_j_dim": dimensionality of the j-th hidden layer output space (decoder)
        activation_fn 
                FC layers activation function [default tf.nn.relu]
        alpha : float
                alpha-beta VAE parameter for weighting the reconstruction loss term [default 1]
        beta : float
                alpha-beta VAE parameter for weighting the KL divergence term [default 1]
        learning_rate : float
                [default 0.001]
        batch_size : int
                batch size of the network [default 32]
        save_path : sring
                path of the model saver [default None] 
        load_model : string 
                path of the model loader [default None]
        max_grad_norm : float
                gradient clipping parameter [default 1]
        dropout : float
                dropout regularization parameter [deafult 0.5]
        is_train : bool
                batch normalization parameter [default True]

        """
        super(DenseCVAE, self).__init__()
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.nn_architecture = nn_architecture
        self.activation_fn = activation_fn
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.dropout = dropout
        self.nn_type = "Dense"
        self.is_train = is_train
        
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
        """Create the Network, define the loss function and the optimizer"""

        # Images and labels placeholders
        self.x = tf.compat.v1.placeholder(tf.float32, shape = [None, self.image_dim], name = "input")
        self.y = tf.compat.v1.placeholder(tf.float32, shape = [None, self.label_dim], name = "label")
        
        # Conditional input
        conditional_input = tf.compat.v1.concat([self.x, self.y], axis = 1)
        
        # Layers initializer
        self.initializer = GlorotUniform()

        #------------------
        # Encoder Network
        #------------------
        with tf.compat.v1.variable_scope("Encoder"):

            self.encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(self.image_dim + self.label_dim,)),
                
                tf.keras.layers.Dense(self.nn_architecture["hidden_enc_1_dim"], kernel_initializer = self.initializer),
                tf.keras.layers.BatchNormalization(trainable = self.is_train),
                tf.keras.layers.Activation(self.activation_fn),
                tf.keras.layers.Dropout(self.dropout),

                tf.keras.layers.Dense(self.nn_architecture["hidden_enc_2_dim"], kernel_initializer = self.initializer),
                tf.keras.layers.BatchNormalization(trainable = self.is_train),
                tf.keras.layers.Activation(self.activation_fn),
                tf.keras.layers.Dropout(self.dropout),

                tf.keras.layers.Dense(self.nn_architecture["hidden_enc_3_dim"], kernel_initializer = self.initializer),
                tf.keras.layers.BatchNormalization(trainable = self.is_train),
                tf.keras.layers.Activation(self.activation_fn),
                tf.keras.layers.Dropout(self.dropout),

                tf.keras.layers.Dense(self.nn_architecture["hidden_enc_4_dim"], kernel_initializer = self.initializer),
                tf.keras.layers.BatchNormalization(trainable = self.is_train),
                tf.keras.layers.Activation(self.activation_fn),
                tf.keras.layers.Dropout(self.dropout),

                # Output layer - No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim)
            ])

        # Mean and log-var of the latent distribution
        self.z_mean, self.z_log_var = tf.split(self.encoder(conditional_input), num_or_size_splits=2, axis=1)

        # Reparametrization trick
        eps = tf.random.normal(shape = (self.batch_size, self.latent_dim), mean = 0.0, stddev = 1.0)       
        self.z = tf.compat.v1.add(self.z_mean, tf.compat.v1.multiply(tf.math.exp(self.z_log_var * .5), eps))
        
        conditional_z = tf.compat.v1.concat([self.z, self.y], axis=1)

        #------------------
        # Decoder Network
        #------------------
        with tf.compat.v1.variable_scope("Decoder"):

            self.decoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim + self.label_dim,)),

                tf.keras.layers.Dense(self.nn_architecture["hidden_dec_1_dim"], kernel_initializer = self.initializer),
                tf.keras.layers.BatchNormalization(trainable = self.is_train),
                tf.keras.layers.Activation(self.activation_fn),
                tf.keras.layers.Dropout(self.dropout),

                tf.keras.layers.Dense(self.nn_architecture["hidden_dec_2_dim"], kernel_initializer = self.initializer),
                tf.keras.layers.BatchNormalization(trainable = self.is_train),
                tf.keras.layers.Activation(self.activation_fn),
                tf.keras.layers.Dropout(self.dropout),

                tf.keras.layers.Dense(self.nn_architecture["hidden_dec_3_dim"], kernel_initializer = self.initializer),
                tf.keras.layers.BatchNormalization(trainable = self.is_train),
                tf.keras.layers.Activation(self.activation_fn),
                tf.keras.layers.Dropout(self.dropout),

                tf.keras.layers.Dense(self.nn_architecture["hidden_dec_4_dim"], kernel_initializer = self.initializer),
                tf.keras.layers.BatchNormalization(trainable = self.is_train),
                tf.keras.layers.Activation(self.activation_fn),
                tf.keras.layers.Dropout(self.dropout),

                # Output layer - No activation
                tf.keras.layers.Dense(self.image_dim)
            ])

        
        logits = self.decoder(conditional_z)

        self.generated_image = tf.nn.sigmoid(logits)


        #------------------
        # Loss Function
        #------------------
        with tf.compat.v1.name_scope('Loss'):
            # KL divergence
            self.latent_loss = self.kl_divergence()
            # Reconstruction loss
            self.reconstr_loss = self.bernoulli_log_likelihood()
            # ELBO Loss
            self.loss = tf.math.scalar_mul(self.alpha, self.reconstr_loss) + tf.math.scalar_mul(self.beta, self.latent_loss)
            # Mean over the batch           
            self.loss = tf.reduce_mean(self.loss)

            
        #------------------
        # Optimizer
        #------------------
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        # Gradient clipping 
        _var = tf.compat.v1.trainable_variables()
        grads, _ = tf.compat.v1.clip_by_global_norm(tf.compat.v1.gradients(self.loss, _var), self.max_grad_norm)
       
        self.train = optimizer.apply_gradients(zip(grads, _var))
                   
        # Saver
        self.saver = tf.compat.v1.train.Saver()


    #---------------------------------
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

    def bernoulli_log_likelihood(self, eps=1e-10):
        """
        Computes reconstruction loss as Bernoulli log likelihood.
        """
        _tmp = self.x * tf.math.log(eps + self.generated_image) + (1 - self.x) * tf.math.log(eps + 1 - self.generated_image)
        return - tf.compat.v1.reduce_sum(_tmp, 1)


    #-------------------------------------------------
    # Reconstruction, Encoding and Decoding methods
    #-------------------------------------------------

    def reconstruct(self, inputs, label):
        """Reconstructs a given data. """
        assert len(inputs) == self.batch_size
        assert len(label) == self.batch_size
        return self.sess.run(self.generated_image, feed_dict={self.x: inputs, self.y: label})

    def encode(self, inputs, label):
        """ Encodes the input into the latent space."""
        return self.sess.run(self.z_mean, feed_dict={self.x: inputs, self.y: label})

    def decode(self, label, z = None):
        """ 
        Generates data starting from the point z in the latent space.
        If z is None, z is drawn from prior in latent space.
        """
        if z is None:
            z = 0.0 + np.random.randn(self.batch_size, self.latent_dim) * 1.0
        return self.sess.run(self.generated_image, feed_dict={self.z: z, self.y: label})