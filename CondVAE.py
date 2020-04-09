import numpy as np 
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform, VarianceScaling

from loss import reconstruction_loss, latent_loss
from utils import dense_layer

tf.compat.v1.disable_eager_execution()

class CVAE (object) :

    def __init__(self, 
                 label_dim,
                 nn_architecture,
                 activation_fn = tf.nn.relu,
                 beta = 1,
                 learning_rate = 0.001,
                 batch_size = 100,
                 save_path = None,
                 load_model = None,
                 max_grad_norm = 1,
                 dropout = 0
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
            activation_fn = tf.nn.relu
            float beta : beta-VAE parameter for weighting the KL divergence [default 1]
            float learning_rate = 0.001
            int batch_size: batch size of the network [1, dataset_dim]
            string save_path = None, 
            string load_model = None, 
            max_grad_norm: maximum norm for the gradients [default 1]
            dropout: dropout regularization parameter [deafult 0]

        """

        self.label_dim = label_dim
        self.nn_architecture = nn_architecture
        self.activation_fn = activation_fn
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.dropout = dropout

        #---------Layer Initializer---------
        if "relu" in self.activation_fn.__name__:
            self.initializer = VarianceScaling()
        if "sigmoid" in self.activation_fn.__name__:
            self.initializer = GlorotUniform()

        #--------Network generation---------
        self._create_network()

        #---------Summary-----------
        tf.compat.v1.summary.scalar("loss", self.loss)

        #--------Session-------------
        self.sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(log_device_placement=False))

        #---------Summary writer for tensor board--------
        self.summary = tf.compat.v1.summary.merge_all()
        if save_path:
            self.writer = tf.compat.v1.summary.FileWriter(save_path, self.sess.graph)
        #---------Load model---------
        if load_model:
            tf.compat.v1.reset_default_graph()
            self.saver.restore(self.sess, load_model)
    
    def _create_network(self):
        """Create the Network and define the Loss function and the Optimizer"""

        #-----------Conditional input---------
        self.x = tf.compat.v1.placeholder(tf.float32, shape = [None, self.nn_architecture["image_size"] *  self.nn_architecture["image_size"] * self.nn_architecture["n_channels"]], name = "input")
        self.y = tf.compat.v1.placeholder(tf.float32, shape = [None, self.label_dim], name = "label")
        _cond_input = tf.compat.v1.concat([self.x, self.y], axis = 1)
        _cond_inpu_dim = self.nn_architecture["image_size"] *  self.nn_architecture["image_size"] * self.nn_architecture["n_channels"] + self.label_dim 
        #----------Encoder Network-----------
        
        with tf.compat.v1.variable_scope("Encoder"):
            
            # 1st hidden layer
            e_layer1 = tf.keras.layers.Dense(self.nn_architecture["hidden_enc_1_dim"],
                                          input_shape = (_cond_inpu_dim,),
                                          activation = self.activation_fn, 
                                          kernel_initializer = self.initializer)(_cond_input)
            if self.dropout:
                e_layer1 = tf.nn.dropout(e_layer1, rate = self.dropout)

            # 2nd hidden layer      
            e_layer2 = tf.keras.layers.Dense(self.nn_architecture["hidden_enc_2_dim"],
                                          input_shape = (self.nn_architecture["hidden_enc_1_dim"],),
                                          activation = self.activation_fn, 
                                          kernel_initializer = self.initializer)(e_layer1)
            if self.dropout:
                e_layer2 = tf.nn.dropout(e_layer2, rate = self.dropout)
            
            # 3rd hidden layer
            e_layer3 = tf.keras.layers.Dense(self.nn_architecture["hidden_enc_3_dim"],
                            input_shape = (self.nn_architecture["hidden_enc_2_dim"],),
                            activation = self.activation_fn, 
                            kernel_initializer = self.initializer)(e_layer2)
            if self.dropout:
                e_layer3 = tf.nn.dropout(e_layer3, rate = self.dropout)

"""
            # 4th hidden layer
            e_layer4 = tf.keras.layers.Dense(self.nn_architecture["hidden_enc_4_dim"],
                input_shape = (self.nn_architecture["hidden_enc_3_dim"],),
                activation = self.activation_fn, 
                kernel_initializer = self.initializer)(e_layer3)

            if self.dropout:
                e_layer4 = tf.nn.dropout(e_layer4, rate = self.dropout)
            """
            # Dense layer to get mean and log(std) of the prior
            self.z_mean = dense_layer(e_layer3, self.nn_architecture["hidden_enc_3_dim"], self.nn_architecture["z_dim"], self.initializer)
            self.z_log_sigma_sq = dense_layer(e_layer3, self.nn_architecture["hidden_enc_3_dim"], self.nn_architecture["z_dim"], self.initializer)

        #------------Reparametrization---------------
        eps = tf.compat.v1.random_normal((self.batch_size, self.nn_architecture["z_dim"]), mean=0, stddev=1, dtype=tf.compat.v1.float32)
        self.z = tf.compat.v1.add(self.z_mean, tf.compat.v1.multiply(tf.math.sqrt(tf.math.exp(self.z_log_sigma_sq)), eps))
        _cond_z = tf.compat.v1.concat([self.z, self.y], axis=1)

        #-----------Decoder Network-------------
        with tf.compat.v1.variable_scope("Decoder"):

            # 1st hidden layer
            d_layer1 = tf.keras.layers.Dense(self.nn_architecture["hidden_dec_1_dim"], 
                                            input_shape = (self.nn_architecture["z_dim"] + self.label_dim,),
                                            activation = self.activation_fn, 
                                            kernel_initializer = self.initializer)(_cond_z)

            if self.dropout:
                d_layer1 = tf.nn.dropout(d_layer1, rate = self.dropout)

            # 2nd hidden layer
            d_layer2 = tf.keras.layers.Dense(self.nn_architecture["hidden_dec_2_dim"], 
                                            input_shape = (self.nn_architecture["hidden_dec_1_dim"],),
                                            activation = self.activation_fn, 
                                            kernel_initializer = self.initializer)(d_layer1)
            if self.dropout:
                d_layer2 = tf.nn.dropout(d_layer2, rate = self.dropout)

            # 3rd hidden layer
            d_layer3 = tf.keras.layers.Dense(self.nn_architecture["hidden_dec_3_dim"], 
                                            input_shape = (self.nn_architecture["hidden_dec_2_dim"],),
                                            activation = self.activation_fn, 
                                            kernel_initializer = self.initializer)(d_layer2)
            if self.dropout:
                d_layer3 = tf.nn.dropout(d_layer3, rate = self.dropout)

			"""
            # 4th hidden layer
            d_layer4 = tf.keras.layers.Dense(self.nn_architecture["hidden_dec_4_dim"], 
                                            input_shape = (self.nn_architecture["hidden_dec_3_dim"],),
                                            activation = self.activation_fn, 
                                            kernel_initializer = self.initializer)(d_layer3)
            if self.dropout:
                d_layer4 = tf.nn.dropout(d_layer4, rate = self.dropout)
            """
            # Output layer
            output = dense_layer(d_layer3, self.nn_architecture["hidden_dec_3_dim"], self.nn_architecture["image_size"] * self.nn_architecture["image_size"] * self.nn_architecture["n_channels"], self.initializer)

            self.x_decoder_mean = tf.nn.sigmoid(output)

        #---------Loss Function-----------
        with tf.compat.v1.name_scope('Loss'):

            self.reconstr_loss = tf.compat.v1.reduce_mean(reconstruction_loss(original = self.x, 
                                                reconstruction = self.x_decoder_mean))
            self.latent_loss = tf.compat.v1.reduce_mean(latent_loss(self.z_mean, self.z_log_sigma_sq))
            self.loss = tf.where(tf.math.is_nan(self.reconstr_loss), 0.0, self.reconstr_loss) + self.beta * tf.where(tf.math.is_nan(self.latent_loss), 0.0, self.latent_loss)

        #-----------Optimizer---------------
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        #maximum gradient norm
        if self.max_grad_norm:
            _var = tf.compat.v1.trainable_variables()
            grads, _ = tf.compat.v1.clip_by_global_norm(tf.compat.v1.gradients(self.loss, _var), self.max_grad_norm)
            self.train = optimizer.apply_gradients(zip(grads, _var))
        else:
            self.train = optimizer.minimize(self.loss)
        # saver
        self.saver = tf.compat.v1.train.Saver()

    #--------Reconstruct, Encode and Decode with the Network----------
    def reconstruct(self, inputs, label):
        """Reconstruct a given data. """
        assert len(inputs) == self.batch_size
        assert len(label) == self.batch_size
        return self.sess.run(self.x_decoder_mean, feed_dict={self.x: inputs, self.y: label})

    def encode(self, inputs, label):
        """ 
        Input -> latent vector 
        """
        return self.sess.run(self.z_mean, feed_dict={self.x: inputs, self.y: label})

    def decode(self, label, z = None, std=0.01, mu = 0):
        """ 
        Generate data by sampling from the latent space.
        If z is None, z is drawn from prior in latent space.
        Otherwise, data for this point in latent space is generated.
        """
        if z is None:
            z = mu + np.random.randn(self.batch_size, self.nn_architecture["z_dim"]) * std 
        return self.sess.run(self.x_decoder_mean, feed_dict={self.z: z, self.y: label})

