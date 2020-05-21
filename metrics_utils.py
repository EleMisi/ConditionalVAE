import cv2
import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

from celeba import CelebA
from ConvolutionalCondVAE import ConvCVAE
from DenseCondVAE import DenseCVAE
from utils import read_VarToSave, batch_generator, get_parameter


def get_activations(model, images):
    """
    Returns the activations for the given set of images.
    """
    assert images.shape == (32,299,299,3), "Wrong images dimension {} (required dimension is {}).".format(images.shape,(32,299,299,3))
    # Pre-processing
    images = images * 255.0
    images = preprocess_input(images, data_format=None)

    # Activations
    activations = model.predict(images)

    return activations

def generation_activations(model, batch_generator, n_batches, inception_model):    
    """
    Returns the list of real images activations and the list of generated images activations.
    The image generation is performed starting from the real images labels.
    """  
    real_activations_list = []
    fake_activations_list = []

    for i in range(n_batches + 1): 
            
            # Real images and generated images
            imgs, labels = next(batch_generator)
            reconstruction = model.decode(labels)
            # Resize images -> (299, 299, 3)
            real_images = resize(imgs, model.nn_type)
            fake_images = resize(reconstruction, model.nn_type)
            #Compute activations
            real_activations_list.append(get_activations(inception_model, real_images))
            fake_activations_list.append(get_activations(inception_model, fake_images))

    assert len(real_activations_list) == len(fake_activations_list)

    return real_activations_list, fake_activations_list



def reconstruction_activations(model, batch_generator, n_batches, inception_model):
    """
    Returns the list of real images activations and the list of reconstrcuted images activations.
    """
    real_activations_list = []
    fake_activations_list = []

    for i in range(n_batches + 1): 
            
            # Real images and reconstructed images
            imgs, labels = next(batch_generator)
            #reconstruction = model(imgs, is_training=False)['x_recon'].numpy()
            #reconstruction = model(imgs)['x_recon'].numpy()
            reconstruction = model.reconstruct(imgs, labels)
            # Resize images -> (299, 299, 3)
            real_images = resize(imgs, 'Conv')
            fake_images = resize(reconstruction, 'Conv')
            # Compute activations
            real_activations_list.append(get_activations(inception_model, real_images))
            fake_activations_list.append(get_activations(inception_model, fake_images))

    assert len(real_activations_list) == len(fake_activations_list)

    return real_activations_list, fake_activations_list



def resize(imgs, model_name):
    """
    Returns the resized images with dimension (299,299,3). 
    """
    images = []

    if model_name == "Dense":
        for img in imgs:
            resized_img = cv2.resize(img.reshape((64,64,3)), dsize=(299, 299), interpolation=cv2.INTER_LINEAR)
            images.append(resized_img)
    else:
        for img in imgs:
            resized_img = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_LINEAR )
            images.append(resized_img)

    assert images[-1].shape == (299,299,3), "Wrong images dimension {} (required dimension is {}).".format(images[-1].shape,(299,299,3))

    return np.array(images)


def load_model(args):
    """
    Loads test labels, sets the required model and creates the batch generator.
    
    Returns
    -------
    model :
        pre-trained ConditionalVAE.
    n_batches : int
        number of batches required to achieves the FID computation sample size.
    batch_gen : 
        batch generator
    """
    
    # Read test_data.pickle 
    test_data = read_VarToSave("./test_data")
    
    pr = "progress-%s-" % args.progress if args.progress else ""

    # Set the model
    if args.neural_network == 'Dense':
        nn_architecture = get_parameter("./parameters.json", args.z_dim)
        acc = np.load("./log/DenseCVAE_%i/%sacc.npz" % (args.z_dim, pr))
        batch_size = acc["batch_size"]
        params = dict(nn_architecture=nn_architecture, batch_size=batch_size)
        params["label_dim"] = test_data["n_attr"]
        params["dropout"] = 0
        params["is_train"] = False
        model = DenseCVAE(load_model="./log/DenseCVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **params)
        print("created Dense")
    else:
        acc = np.load("./log/ConvCVAE_%i/%sacc.npz" % (args.z_dim, pr))
        batch_size = acc["batch_size"]
        params = dict(batch_size=batch_size)
        params["label_dim"] = test_data["n_attr"]
        params["latent_dim"] = args.z_dim
        params["is_train"] = False
        model = ConvCVAE(load_model="./log/ConvCVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **params)

    
    n_batches = int(args.sample_size / batch_size)
    batch_gen = batch_generator(batch_size, test_data['test_labels'], model_name = model.nn_type)
    print("\nModel loaded.\n")

    return model, n_batches, batch_gen

