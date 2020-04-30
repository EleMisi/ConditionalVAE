import sys
import os
import argparse
import json

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf

from celeba import CelebA
from DenseCondVAE import DenseCVAE
from ConvolutionalCondVAE import ConvCVAE
from utils import read_VarToSave, batch_generator, get_parameter



def image_generation(model, test_data, name = None, target_attr = None, std=0.01):
    """
    Generates and plots 16 images with a given attribute (if given).
    - list target_attr : list of desidered attributes [default None]
    """
    # Vector of user-defined attributes.
    if target_attr:       
        attr_vect = np.zeros(test_data["n_attr"])
        for attr in target_attr:
            attr_vect[attr] = 1
        labels = np.tile(attr_vect, reps = [model.batch_size, 1])
        name = name + "Attributes"+str(target_attr)
        print("Generation of 16 images with attributes: ", target_attr )

   # Vector of attributes taken from the test set.
    else:        
        batch_gen = batch_generator(test_data['batch_dim'], test_data['test_labels'], model_name = model.nn_type)
        _, labels = next(batch_gen)
        name = name + "_Test_set_Attr"
        print("Generation of 16 images with fixed attributes.")

    generated = model.decode(label = labels)
    
    # Plot
    imshow_grid(generated, model_name = model.nn_type, shape=[4, 4], name = name, save = True)


def image_reconstruction(model, test_data, save_path=None):
    """
    Reconstructs and plots 5 test images.
    """
    batch_gen = batch_generator(test_data['batch_dim'], test_data['test_labels'], model_name = model.nn_type)
    _x, _y = next(batch_gen)
    reconstruction = model.reconstruct(_x, _y)

    # Plot
    plt.figure(figsize=(6, 10))
    if model.nn_type == "Dense":
        for i in range(5):
            plt.subplot(5, 2, 2 * i + 1)
            plt.imshow(_x[i].reshape(64, 64, 3))
            plt.title("Original")
            plt.subplot(5, 2, 2 * i + 2)
            plt.imshow(reconstruction[i].reshape(64, 64, 3))
            plt.title("Reconstruction")
            plt.tight_layout()
        if save_path:
            plt.savefig(save_path + "reconstruction.png", bbox_inches="tight")
            plt.clf()
        else:
            plt.show()

    else:
        for i in range(5):
            plt.subplot(5, 2, 2 * i + 1)
            plt.imshow(_x[i].reshape(64, 64, 3))
            plt.title("Original")
            plt.subplot(5, 2, 2 * i + 2)
            plt.imshow(reconstruction[i])
            plt.title("Reconstruction")
            plt.tight_layout()
        if save_path:
            plt.savefig(save_path + "reconstruction.png", bbox_inches="tight")
            plt.clf()
        else:
            plt.show()


    print("Reconstruction of 5 images from the test set.")

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



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Conditional VAE plot script.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--plot_type', action='store', nargs='?', const=None, default=None, type=str,
                        choices=None, metavar=None, help="""Plot type.\n- reconstr: reconstruction - gen: generate 10 images with random/given attributes. [default None (all)] """)
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Latent dimension. [default: 20]', metavar=None)
    parser.add_argument('-nn', '--neural_network', action='store', nargs='?', const=None, default='Conv', type=str,
                        choices=None, help='Neural network architecture. [default: Conv]', metavar=None)
    parser.add_argument('-p', '--progress', action='store', nargs='?', const=None, default=None, type=str, metavar=None,
                        choices=None, help='Use progress model (model is saved each 50 epoch). [default: None]')
    parser.add_argument('-s', '--std', action='store', nargs='?', const=None, default=0.1, type=float,
                        choices=None, help='Std of gaussian noise for `gen_rand` plotting. [default: 0.1]', metavar=None)
    parser.add_argument('-t', '--target', action='store', nargs='?', const=None, default=False, type=bool, metavar=None,
                        choices=None,
                        help='Target attribute(s). [default: False (test set attributes)]')
    args = parser.parse_args()


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
    else:
        acc = np.load("./log/ConvCVAE_%i/%sacc.npz" % (args.z_dim, pr))
        params = dict(batch_size=acc["batch_size"])
        params["label_dim"] = test_data["n_attr"]
        params["latent_dim"] = args.z_dim
        params["dropout"] = 0
        params["is_train"] = False
        model = ConvCVAE(load_model="./log/ConvCVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **params)

    
    # Saving path for images
    folder = "./results/"
    if not os.path.exists(folder):
        os.mkdir(folder)


    # Reconstruction
    if args.plot_type is None or args.plot_type == "reconstr":
        image_reconstruction(model, test_data, save_path=folder) 
    
    # Generation 
    if args.plot_type is None or args.plot_type == "gen":
        target_attr = None
        if args.target :
            n = input("Number of desired attributes? \n (type help for instructions)\n")
            if n == "help":
                print("The attributes must be insert as a list of integers: 1 3 15 35","Here is the encoding:", sep = '\n')
                for a, i in zip(test_data['attr'], range(test_data['n_attr'])):
                    print(a, "-->", i)
                n = int(input("Number of desired attributes?\n"))
            else:
                n = int(n)

            target_attr = list(map(int,input("\nEnter the desired attributes:\n").strip().split()))[:n] 

        image_generation(model, test_data, name = folder + "random_generation", target_attr = target_attr, std=args.std)


