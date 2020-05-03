import argparse
import cv2
import json
import numpy as np
import os

from celeba import CelebA
from ConvolutionalCondVAE import ConvCVAE
from DenseCondVAE import DenseCVAE
from utils import batch_generator, get_parameter, read_VarToSave, save_VarToSave


#@TODO: implement the same operations for image generation.


def pre_processing(imgs, model_name):
    """
    Resizes the images to get the required dimension for Inception_v3: (299,299,3) 
    """
    n_imgs = len(imgs)
    images = []

    if model_name == "Dense":
        for img in imgs:
            images.append(img.reshape((64,64,3)))
    else:
      images = imgs

    for i in range(n_imgs):
        images[i] = cv2.resize(images[i], dsize=(299, 299), interpolation=cv2.INTER_CUBIC)

    assert images[-1].shape == (299,299,3)
    return images


def build_sample(model, batch_generator):
    """
    Returns a sample of test images and a sample reconstructed images.
    The reconstruction is performed by the given model.
    """
    imgs, labels = next(batch_generator)
    reconstruction = model.reconstruct(imgs, labels)

    original_imgs = pre_processing(imgs, model.nn_type)
    reconstr_imgs = pre_processing(reconstruction, model.nn_type)

    return original_imgs, reconstr_imgs

def prepare_images(args, save_path):
    """
    Loads the test images, generates the reconstructed images and pre-processes them.
    The images are stored in save_path.
    """

    # Read test_data.pickle 
    test_data = read_VarToSave("./test_data")
    
    pr = "progress-%s-" % args.progress if args.progress else ""

    # Set the model
    if args.neural_network == 'Dense':
        nn_architecture = get_parameter("./parameters.json", args.z_dim)
        acc = np.load("./log/DenseCVAE_%i/%sacc.npz" % (args.z_dim, pr))
        params = dict(nn_architecture=nn_architecture, batch_size=acc["batch_size"])
        params["label_dim"] = test_data["n_attr"]
        params["dropout"] = 0
        params["is_train"] = False
        model = DenseCVAE(load_model="./log/DenseCVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **params)
        print("created Dense")
    else:
        acc = np.load("./log/ConvCVAE_%i/%sacc.npz" % (args.z_dim, pr))
        params = dict(batch_size=acc["batch_size"])
        params["label_dim"] = test_data["n_attr"]
        params["latent_dim"] = args.z_dim
        params["dropout"] = 0
        params["is_train"] = False
        model = ConvCVAE(load_model="./log/ConvCVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **params)

    batch_size = params["batch_size"]
    n_batches = int(args.sample_size / batch_size)
    batch_gen = batch_generator(batch_size, test_data['test_labels'], model_name = model.nn_type)

    # Reconstruction, pre-processing and saving procedure.
    for i in range(n_batches + 1): 
        original_imgs, reconstructed_imgs = build_sample(model, batch_gen) 

        save_VarToSave(save_path + 'original_images_' + str(i), original_imgs)
        save_VarToSave(save_path + 'reconstr_images_' + str(i), reconstructed_imgs)
        print("Loaded {}° batch".format(i))

    print("Saved {} images".format((n_batches + 1)*batch_size))

    return batch_size


def load_images(args, batch_size, load_path):
    """
    Returns the list of real images and the list of reconstructed images loaded from load_path.
    """

    n_batches = int(args.sample_size / batch_size)

    original_imgs = []
    reconstr_imgs = []

    for i in range(n_batches + 1): 
        original = read_VarToSave(load_path + 'original_images_' + str(i))
        reconstructed = read_VarToSave(load_path + 'reconstr_images_' + str(i))
        original_imgs.append(original)
        reconstr_imgs.append(reconstructed)
        print("Loading {}° batch".format(i))

    original_imgs = [img for batch in original_imgs for img in batch][:args.sample_size]
    reconstr_imgs = [img for batch in reconstr_imgs for img in batch][:args.sample_size]

    assert len(original_imgs) == args.sample_size
    assert len(reconstr_imgs) == args.sample_size
    assert original_imgs[0].shape == (299, 299, 3)
    assert reconstr_imgs[0].shape == (299, 299, 3)

    print("\nDone.")

    return original_imgs, reconstr_imgs 



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image preprocessing script for FID computation.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--mode', action='store', nargs='?', const=None, default=None, type=str,
                        choices=None, metavar=None, help='Mode.\n- reconstr: reconstruction - gen: generation. [default None (both)]')
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Latent dimension. [default: 20]', metavar=None)
    parser.add_argument('-nn', '--neural_network', action='store', nargs='?', const=None, default='Conv', type=str,
                        choices=None, help='Neural network architecture. [default: Conv]', metavar=None)
    parser.add_argument('-p', '--progress', action='store', nargs='?', const=None, default=None, type=str, metavar=None,
                        choices=None, help='Use progress model (model is saved each 50 epoch). [default: None]')
    parser.add_argument('-s', '--sample_size', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Sample size fot FID computation. [default: 100]', metavar=None) 
    args = parser.parse_args()

    save_path = '/FID/images/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create the sample of real and reconstructed images
    batch_size = prepare_images(args, save_path)
    
    # Load real and reconstructed images
    real_imgs, reconstr_imgs = load_images(args, batch_size, save_path)


