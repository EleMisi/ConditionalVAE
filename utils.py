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

#----------CELEBA DATASET UTILS------------

#---PROVVISORIAL----
def load_data():
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
                break

    with (open(path_arr, "rb")) as openfile:
        while True:
            try:
                imgs.append(pickle.load(openfile))
            except EOFError:
              break
      
    for k in imgs[0].keys():
        id_images.append(k)

    return label, imgs, id_images

def prepare_dataset():
    """
    Load the dataset and return a dictionary with each image and its label.
    """
    label, imgs, id_imgs = load_data() #Load dataset with labels

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
    
    np.random.seed(42)
    # Load data
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

    # Session initialization
    model.sess.run(tf.compat.v1.global_variables_initializer())

    #-----------------Train-----------------

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

        #---------------Validation--------------

        _results = np.mean(_results, 0)
        logger.info("epoch %i: loss %0.3f, reconstr. loss %0.3f, latent loss %0.3f"
                    % (_e, _results[0], _results[1], _results[2]))

        results.append(_results)
        if (_e + 1) % 10 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(results),
                     learning_rate=model.learning_rate, epoch=epoch, batch_size=model.batch_size,
                     clip=model.max_grad_norm)

    #Save the model                 
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/acc.npz" % save_path, loss=np.array(results), learning_rate=model.learning_rate, epoch=epoch,
             batch_size=model.batch_size, clip=model.max_grad_norm)
