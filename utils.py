import logging
import os
import numpy as np
import tensorflow as tf
import sys


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

#--------MNIST DATASET UTILS-----------

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
    n_iter = int(n / model.batch_size)
    # logger
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = create_log(save_path+"log")
    logger.info("train: data size(%i), batch num(%i), batch size(%i)" % (n, n_iter, model.batch_size))
    result = []
    # Initializing the tensor flow variables
    model.sess.run(tf.global_variables_initializer())
    for _e in range(epoch):
        _result = []
        for _b in range(n_iter):
            # train
            _x, _y = data.train.next_batch(model.batch_size)
            _x = shape_2d(_x, model.batch_size) if input_image else _x

            if mode in ["conditional", "unsupervised"]:  # conditional unsupervised model
                feed_val = [model.summary, model.loss, model.reconstr_loss , model.latent_loss, model.train]
                feed_dict = {model.x: _x, model.y: _y} if mode == "conditional" else {model.x: _x}
                summary, loss, reconstr_loss , latent_loss, _ = model.sess.run(feed_val, feed_dict=feed_dict)
                __result = [loss, reconstr_loss , latent_loss]
            elif mode == "supervised":  # supervised model
                feed_val = [model.summary, model.loss, model.accuracy, model.train]
                feed_dict = {model.x: _x, model.y: _y, model.is_training: True}
                summary, loss, acc, _ = model.sess.run(feed_val, feed_dict=feed_dict)
                __result = [loss, acc]
            else:
                sys.exit("unknown mode !")
            _result.append(__result)
            model.writer.add_summary(summary, int(_b + _e * model.batch_size))

        # validation
        if mode == "supervised":  # supervised model
            _x = shape_2d(data.test.images, data.test.num_examples) if input_image else data.test.image
            _y = data.test.labels
            feed_dict = {model.x: _x, model.y: _y, model.is_training: False}
            loss, acc = model.sess.run([model.loss, model.accuracy], feed_dict=feed_dict)
            _result = np.append(np.mean(_result, 0), [loss, acc])
            logger.info("epoch %i: acc %0.3f, loss %0.3f, train acc %0.3f, train loss %0.3f"
                        % (_e, acc, loss, _result[1], _result[0]))
        else:
            _result = np.mean(_result, 0)
            logger.info("epoch %i: loss %0.3f, reconstr. loss %0.3f, latent loss %0.3f"
                        % (_e, _result[0], _result[1], _result[2]))

        result.append(_result)
        if _e % 50 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(result),
                     learning_rate=model.learning_rate, epoch=epoch, batch_size=model.batch_size,
                     clip=model.max_grad_norm)
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/acc.npz" % save_path, loss=np.array(result), learning_rate=model.learning_rate, epoch=epoch,
             batch_size=model.batch_size, clip=model.max_grad_norm)



def full_connected(x, weight_shape, initializer):
    """ fully connected layer
    - weight_shape: input size, output size
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.matmul(x, weight), bias)




def celebA_train():
    pass
