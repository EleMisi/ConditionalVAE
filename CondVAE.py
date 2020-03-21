import tensorflow as tf
import numpy as np 
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
from loss import reconstruction_loss, latent_loss


class CVAE (object) :

    def __init__(self, label_dim,
                 nn_architecture,
                 activation_fn = tf.nn.sigmoid,
                 learning_rate = 0.001,
                 batch_size = 100,
                 save_path = None,
                 load_model = None,
                 max_grad_norm = 1
                 ):

        """
        Parameters:
            int label_dim: label vector dimension 
            dict nn_architecture: 
                    input_dim: input dimension
                    z_dim: latent space dimension
                    dataset_dim: traing set dimension
                    hidden_enc_1_dim: dimensionality of the 1st hidden layer output space (encoder)
                    hidden_enc_2_dim: dimensionality of the 2nd hidden layer output space (encoder)
                    hidden_dec_1_dim: dimensionality of the 1st hidden layer output space (decoder)
                    hidden_dec_2_dim: dimensionality of the 2nd hidden layer output space (decoder)
            activation_fn: activation function
            float learning_rate: default 0.001
            int batch_size: batch size of the network [1, dataset_dim]
            save_path = None, ????????
            load_model = None, ??????
            max_grad_norm: maximum norm for the gradients (for controlling the overfitting) ???

        """

        self.label_dim = label_dim
        self.nn_architecture = nn_architecture
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        #---------Layer Initializer---------
        if "relu" in self.activation_fn.__name__:
            self.initializer = variance_scaling_initializer()
        else:
            self.initializer = xavier_initializer()

        #--------Network generation---------
        self._create_network()

        #---------Summary-----------
        tf.summary.scalar("loss", self.loss)

        #--------Session-------------
        self.session = tf.Session(config = tf.ConfigProto(log_device_placement=False))

        #---------Summary writer for tensor board--------
        self.summary = tf.summary.merge_all()
        if save_path:
            self.writer = tf.summary.FileWriter(save_path, self.session.graph)
        #---------Load model---------
        if load_model:
            tf.reset_default_graph()
            self.saver.restore(self.session, load_model)
    
    def _create_network(self):
        """Create the Network and define the Loss function and the Optimizer"""

        #-----------Conditional input---------
        self.x = tf.placeholder(tf.float32, shape = [None, self.nn_architecture["input_dim"]], name = "input")
        self.y = tf.placeholder(tf.float32, shape = [None, self.label_dim], name = "label")
        _cond_input = tf.concat([self.x, self.y], axis = 1)

        #----------Encoder Network-----------
        # input (1d vector) -> FC x 3 -> latent
        with tf.variable_scope("Encoder"):
            
            _output1 = tf.keras.layers.Dense(self.nn_architecture["hidden_enc_1_dim"],
                                          input_shape = (self.nn_architecture["input_dim"] + self.label_dim,),
                                          activation = self.activation_fn, 
                                          kernel_initializer = self.initializer)(_cond_input)
            
            _output2 = tf.keras.layers.Dense(self.nn_architecture["hidden_enc_2_dim"],
                                          input_shape = (self.nn_architecture["hidden_enc_1_dim"],),
                                          activation = self.activation_fn, 
                                          kernel_initializer = self.initializer)(_output1)

            self.z_mean = tf.keras.layers.Dense(self.nn_architecture["z_dim"],
                                            input_shape = (self.nn_architecture["hidden_enc_2_dim"],),
                                            activation = self.activation_fn, 
                                            kernel_initializer = self.initializer)(_output2)

            self.z_log_sigma_sq = tf.keras.layers.Dense(self.nn_architecture["z_dim"],
                                                    input_shape = (self.nn_architecture["hidden_enc_2_dim"],),
                                                    activation = self.activation_fn, 
                                                    kernel_initializer = self.initializer)(_output2)

        #------------Reparametrization---------------
        eps = tf.random_normal((self.batch_size, self.nn_architecture["z_dim"]), mean=0, stddev=1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        _cond_z = tf.concat([self.z, self.y], axis=1)

        #-----------Decoder Network-------------
        with tf.variable_scope("Decoder"):

            _output1 = tf.keras.layers.Dense(self.nn_architecture["hidden_dec_1_dim"], 
                                            input_shape = (self.nn_architecture["z_dim"] + self.label_dim,),
                                            activation = self.activation_fn, 
                                            kernel_initializer = self.initializer)(_cond_z)

            _output2 = tf.keras.layers.Dense(self.nn_architecture["hidden_dec_2_dim"], 
                                            input_shape = (self.nn_architecture["hidden_dec_1_dim"],),
                                            activation = self.activation_fn, 
                                            kernel_initializer = self.initializer)(_output1)
            
            _output = tf.keras.layers.Dense(self.nn_architecture["input_dim"], 
                                            input_shape = (self.nn_architecture["hidden_dec_1_dim"],),
                                            activation = self.activation_fn, 
                                            kernel_initializer = self.initializer)(_output2)

            self.x_decoder_mean = tf.nn.sigmoid(_output)

        #---------Loss Function-----------
        with tf.name_scope('Loss'):

            self.reconstr_loss = tf.reduce_mean(reconstruction_loss(original = self.x, 
                                                reconstruction = self.x_decoder_mean))
            self.latent_loss = tf.reduce_mean(latent_loss(self.z_mean, self.z_log_sigma_sq))
            self.loss = (tf.where(tf.is_nan(self.reconstr_loss), 0.0, self.reconstr_loss) + 
                            tf.where(tf.is_nan(self.latent_loss), 0.0, self.latent_loss))

        #-----------Optimizer---------------
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        #maximum gradient norm
        if self.max_grad_norm:
            _var = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, _var), self.max_grad_norm)
            self.train = optimizer.apply_gradients(zip(grads, _var))
        else:
            self.train = optimizer.minimize(self.loss)
        # saver
        self.saver = tf.train.Saver()

    #--------Reconstruct, Encode and Decode with the Network----------
    def reconstruct(self, inputs, label):
        """Reconstruct a given data. """
        assert len(inputs) == self.batch_size
        assert len(label) == self.batch_size
        return self.session.run(self.x_decoder_mean, feed_dict={self.x: inputs, self.y: label})

    def encode(self, inputs, label):
        """ Embed given data to latent vector. """
        return self.session.run(self.z_mean, feed_dict={self.x: inputs, self.y: label})

    def decode(self, label, z=None, std=0.01, mu=0):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is generated.
        Otherwise, z_mu is drawn from prior in latent space.
        """
        z = mu + np.random.randn(self.batch_size, self.nn_architecture["n_z"]) * std if z is None else z
        return self.session.run(self.x_decoder_mean, feed_dict={self.z: z, self.y: label})