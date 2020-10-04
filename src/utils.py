import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import pickle
import numpy as np
import tensorflow as tf



#######################
# Train Step Function #
#######################


def train_step(data, model, optimizer):


    with tf.GradientTape() as tape:

        model_output = model(data, is_train = True)

    trainable_variables = model.trainable_variables
    grads = tape.gradient(model_output['loss'], trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

    total_loss = model_output['loss'].numpy().mean()
    recon_loss = model_output['reconstr_loss'].numpy().mean()
    latent_loss = model_output['latent_loss'].numpy().mean()

    return total_loss, recon_loss, latent_loss




##################################
# Encoding and Decoding methods  #
##################################


def encode(self, inputs, label):
    """ Encodes the input into the latent space."""
    return self.sess.run(self.z_mean, feed_dict={self.x: inputs, self.y: label})


def decode(self, label, z = None):
    """ 
    Generates data starting from the point z in the latent space.
    If z is None, z is drawn from prior in latent space.
    """
    if z is None:
        z = 0.0 + np.random.randn(self.batch_size, self.latent_dim) * 0.75
    return self.sess.run(self.generated_image, feed_dict={self.z_sample_3: z, self.y: label})




########################
#  Utils for plotting  #
########################


def batch_generator(batch_dim, test_labels, model_name):
    """
    Batch generator using the given list of labels.
    """
    while True:
        batch_imgs = []
        labels = []
        for label in (test_labels):
            labels.append(label)
            if len(labels) == batch_dim:
                batch_imgs = create_image_batch(labels, model_name)
                batch_labels = [x[1] for x in labels]
                yield np.asarray(batch_imgs), np.asarray(batch_labels)
                batch_imgs = []
                labels = []
                batch_labels = []
        if batch_imgs:
            yield np.asarray(batch_imgs), np.asarray(batch_labels)


def get_image(image_path, model_name, img_size = 128, img_resize = 64, x = 25, y = 45):
    """
    Crops, resizes and normalizes the target image.
        - If model_name == Dense, the image is returned as a flattened numpy array with dim (64*64*3)
        - otherwise, the image is returned as a numpy array with dim (64,64,3)
    """

    img = cv2.imread(image_path)
    img = img[y:y+img_size, x:x+img_size]
    img = cv2.resize(img, (img_resize, img_resize))
    img = np.array(img, dtype='float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img /= 255.0 # Normalization to [0.,1.]

    if model_name == "Dense" :
        img = img.ravel()
    
    return img


def create_image_batch(labels, model_name):
    """
    Returns the list of images corresponding to the given labels.
    """
    imgs = []
    imgs_id = [item[0] for item in labels]

    for i in imgs_id:
        image_path ='/input/CelebA/img_align_celeba/img_align_celeba/' + i
        imgs.append(get_image(image_path, model_name))

    return imgs



def convert_batch_to_image_grid(image_batch, dim = 64):
    reshaped = (image_batch.reshape(4, 8, dim, dim, 3)
              .transpose(0, 2, 1, 3, 4)
              .reshape(4 * dim, 8 * dim, 3))
    return reshaped 


def imshow_grid(imgs, model_name, shape=[2, 5], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
    size = shape[0] * shape[1]
    if model_name == "Dense":
        for i in range(size):
            grid[i].axis('off')
            grid[i].imshow(imgs[i].reshape(64, 64, 3))  
        if save:
            plt.savefig(str(name) + '.png')
            plt.clf()
        else:
            plt.show()
    else:
        for i in range(size):
            grid[i].axis('off')
            grid[i].imshow(imgs[i])  
        if save:
            plt.savefig(str(name) + '.png')
            plt.clf()
        else:
            plt.show()


##########################################
#   Utils to save and read pickle files  #
##########################################

def save_data(file_name, data):
    """
    Saves data on file_name.pickle.
    """
    with open((file_name+'.pickle'), 'wb') as openfile:
        print(type(data))
        pickle.dump(data, openfile)


def read_data(file_name):
    """
    Reads file_name.pickle and returns its content.
    """
    with (open((file_name+'.pickle'), "rb")) as openfile:
        while True:
            try:
                objects=pickle.load(openfile)
            except EOFError:
                break
    return objects