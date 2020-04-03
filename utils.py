import logging
import os
import json

import cv2
import pickle
import numpy as np
import tensorflow as tf
import sys

from celeba import CelebA


#---Fully connected layer with no activation------

def fully_conn(x, weight_shape, initializer):
    """ Fully connected layer.
    - weight_shape: input size, output size
    - initializer: tf function
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.matmul(x, weight), bias)

#-----------Get paramters---------
def get_parameter(path, z_dim):
    with open(path) as f:
        p = json.load(f)
    if z_dim:
        p["z_dim"] = z_dim
    return p

#---------Log util--------------
def create_log(name):
    """Logging."""
    if os.path.exists(name):
        os.remove(name)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    # handler for log file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter("H_1, %(asctime)s %(levelname)8s %(message)s"))

    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter("H_2, %(asctime)s %(levelname)8s %(message)s"))
    log.addHandler(handler1)
    log.addHandler(handler2)
    return log


#------------------------------------
#------------CelebA Train------------
#------------------------------------


def celebA_train(model, dataset, epoch, save_path="./"):

    n_train = len(dataset.train_set)
    n_batches = int(n_train / model.batch_size)

    # Log
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log = create_log(save_path+"Log")
    log.info("Training set dimension(%i) batch num(%i), batch size(%i)" % (n_train, n_batches, model.batch_size))
    results = []

    # Session initialization
    model.sess.run(tf.compat.v1.global_variables_initializer())

    #-----------------Train-----------------

    for _e in range(epoch):
        _results = []
        for _b in range(n_batches):
            _x, _y = dataset.next_batch(model.batch_size, dataset.train_set, dataset.train_labels)
            feed_val = [model.summary, model.loss, model.reconstr_loss , model.latent_loss, model.train]
            feed_dict = {model.x: _x, model.y: _y} 
            summary, loss, reconstr_loss , latent_loss, _ = model.sess.run(feed_val, feed_dict=feed_dict)
            __result = [loss, reconstr_loss , latent_loss]
        
            _results.append(__result)
            model.writer.add_summary(summary, int(_b + _e * model.batch_size))
            _b += 1

        #---------------Validation--------------

        _results = np.mean(_results, 0)
        log.info("epoch %i: loss %0.3f, reconstr. loss %0.3f, latent loss %0.3f"
                    % (_e, _results[0], _results[1], _results[2]))    
        results.append(_results)

        #----------Save progress every 5 epochs----------
        if (_e + 1) % 5 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(results),
                     learning_rate=model.learning_rate, epoch=epoch, batch_size=model.batch_size,
                     clip=model.max_grad_norm)

    #------------Save the final model--------------                 
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/acc.npz" % save_path, loss=np.array(results), learning_rate=model.learning_rate, epoch=epoch,
             batch_size=model.batch_size, clip=model.max_grad_norm)

