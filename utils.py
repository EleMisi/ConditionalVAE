import logging
import os
import glob #to iterate in a folder
import cv2
import pickle
import numpy as np
import tensorflow as tf
import sys


#---FULLY CONNECTED LAYER - NO ACTIVATION------

def FC(x, weight_shape, initializer):
    """ fully connected layer
    - weight_shape: input size, output size
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.matmul(x, weight), bias)

#-----------------------------

def create_log(name):
    """Logging."""
    if os.path.exists(name):
        os.remove(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # handler for logger file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

#--------MNIST DATASET---------

def mnist_loader():
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    mnist = read_data_sets('MNIST_data', one_hot=True)
    n_sample = mnist.train.num_examples
    return mnist, n_sample


def shape_2d(_x, batch_size):
    _x = _x.reshape(batch_size, 28, 28)
    return np.expand_dims(_x, 3)


def mnist_train(model, epoch, save_path="./", mode="supervised", input_image=False):
    """ Train model based on mini-batch of input data.

    :param model:
    :param epoch:
    :param save_path:
    :param mode: conditional, supervised, unsupervised
    :param input_image: True if use CNN for top of the model
    :return:
    """
    # load mnist
    data, n = mnist_loader()
    n_iter = int(n / model.batch_size) #number of minibatches
    # logger
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = create_log(save_path+"log")
    logger.info("train: data size(%i), batch num(%i), batch size(%i)" % (n, n_iter, model.batch_size))
    result = []

    # Initializing the tensor flow variables
    model.sess.run(tf.global_variables_initializer())
    
    #Start training
    for _e in range(epoch):
        _result = []
        for _b in range(n_iter):
            # train
            _x, _y = data.train.next_batch(model.batch_size)
            _x = shape_2d(_x, model.batch_size) if input_image else _x

            
            feed_val = [model.summary, model.loss, model.reconstr_loss , model.latent_loss, model.train]
            feed_dict = {model.x: _x, model.y: _y} 
            summary, loss, reconstr_loss , latent_loss, _ = model.sess.run(feed_val, feed_dict=feed_dict)
            __result = [loss, reconstr_loss , latent_loss]
        
            _result.append(__result)
            model.writer.add_summary(summary, int(_b + _e * model.batch_size))

        # Validation
        _result = np.mean(_result, 0)
        logger.info("epoch %i: loss %0.3f, reconstr. loss %0.3f, latent loss %0.3f"
                    % (_e, _result[0], _result[1], _result[2]))

        result.append(_result)
        if _e % 50 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(result),
                     learning_rate=model.learning_rate, epoch=epoch, batch_size=model.batch_size,
                     clip=model.max_grad_norm)

    #Save the model                 
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/acc.npz" % save_path, loss=np.array(result), learning_rate=model.learning_rate, epoch=epoch,
             batch_size=model.batch_size, clip=model.max_grad_norm)


#----------CELEBA DATASET UTILS------------

#Data must be a list of listsr a matrix
def load_CelebA():
    """
    data = np.array([
           [ ["rosso1"], ["verde1"], ["blu1"] ],
           [ ["rosso2"], ["verde2"], ["blu2"] ],
           [ ["rosso3"], ["verde3"], ["blu3"] ],
           [ ["rosso4"], ["verde4"], ["blu4"] ]
           ])

    N = 4
    """ 
    data = []
    images = glob.glob("images/*.jpg")
    for image in images:
        img = cv2.imread(image)
        data.append(img)

    return np.array(data), len(data)

def create_batch(data, n_samples, batch_size):
    assert batch_size % 2 == 0
    n_batch = int(n_samples / batch_size) 
    batches = {}
    _i = 0
    for _b in range(n_batch):
        _n = _i + batch_size
        #print(_i, _n)
        batches["Batch %i" % _b] = data[_i : _n]        
        _i = _n
    return batches, n_batch        

#---ROVVISORIAL----
def load_todos():
    label = []
    imgs = []
    id_images = []
    path_arr = "./ProvvisorialData/arr_no_Eyeglasses.pickle"
    path_label = "./ProvvisorialData/img_no_Eyeglasses.pickle"

    with (open(path_label, "rb")) as openfile:
        while True:
            try:
                label.append(pickle.load(openfile))
            except EOFError:
                #print("here1")
                break

    with (open(path_arr, "rb")) as openfile:
        while True:
            try:
                imgs.append(pickle.load(openfile))
            except EOFError:
              #print("here2")
              break
      
    for k in imgs[0].keys():
        id_images.append(k)

    return label, imgs, id_images

def prepare_dataset():
    """
    Load the dataset and return a dictionary with each image and its label.
    """
    label, imgs, id_imgs = load_todos() #Load dataset with labels

    data_label_dict = {}
    for k in id_imgs:
        for j in range(len(label[0])):
            if label[0][j][0] == k:
                im = imgs[0][k].ravel() 
                im_norm = im / 255.
                data_label_dict[k] = [im_norm, label[0][j][1]]

    for k,v in data_label_dict.items():
      v[1] = [v[1]['Eyeglasses'], v[1]['Blond_Hair']]

    return data_label_dict

def prepare_batches_provv(data, n_samples, batch_size):

    """
    sputa fuori un dizionario con keys i singoli batch 
    e come items le immagini [0] e le relative labels [1]
    batches['Batch 0'][9][0] -> decima immagine del primo batch ravellata
    batches['Batch 0'][9][1] -> decimo vettore di label del primo batch
    """
    assert batch_size % 2 == 0
    n_batch = int(n_samples / batch_size)
    batches = {}
    _i = 0
    imgs = []
    lbl = []
    for v in data:
        imgs.append(v[0])
        lbl.append(v[1])
    for _b in range(n_batch):
        _n = _i + batch_size
        print(_i, _n)
        batches["Batch %i" % _b] =[imgs[_i : _n], lbl[_i : _n]]       
        _i = _n
    return batches, n_batch   

#-----attempt 3---------------

def data_and_labels(data_label_dict):
    imgs = []
    label = []
    for v in data_label_dict.values():
        imgs.append(v[0])
        label.append(v[1])
    return imgs, label

def next_batch(batch_dim, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_dim]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#-----------------------------------------------

def celebA_train(model, epoch, save_path="./", input_image=False):
    
    """
    #Load data
    data, N = load_CelebA()
    label = load_label(model.batch_size)
    batches, n_batch = create_batch(data, N, model.batch_size)
    """
    dataset_dict = prepare_dataset()
    n_samples = len(dataset_dict)
    imgs, labels = data_and_labels(dataset_dict)
    n_batches = int(n_samples / model.batch_size) #batch_size must be a divisor of n_samples
    
    # logger
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = create_log(save_path+"log")
    logger.info("train: data size(%i), batch num(%i), batch size(%i)" % (n_samples, n_batches, model.batch_size))
    results = []

    #Session initialization
    model.sess.run(tf.global_variables_initializer())

    #Train
    
    for _e in range(epoch):
        _results = []
        for _b in range(n_batches):
            _x, _y = next_batch(model.batch_size, imgs, labels)
            feed_val = [model.summary, model.loss, model.reconstr_loss , model.latent_loss, model.train]
            feed_dict = {model.x: _x, model.y: _y} 
            summary, loss, reconstr_loss , latent_loss, _ = model.sess.run(feed_val, feed_dict=feed_dict)
            __result = [loss, reconstr_loss , latent_loss]
        
            _results.append(__result)
            model.writer.add_summary(summary, int(_b + _e * model.batch_size))
            _b += 1

        # Validation
        _results = np.mean(_results, 0)
        logger.info("epoch %i: loss %0.3f, reconstr. loss %0.3f, latent loss %0.3f"
                    % (_e, _results[0], _results[1], _results[2]))

        results.append(_results)
        if _e % 50 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(results),
                     learning_rate=model.learning_rate, epoch=epoch, batch_size=model.batch_size,
                     clip=model.max_grad_norm)

    #Save the model                 
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/acc.npz" % save_path, loss=np.array(results), learning_rate=model.learning_rate, epoch=epoch,
             batch_size=model.batch_size, clip=model.max_grad_norm)
