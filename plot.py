import sys
import os
import argparse
import json

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import tensorflow as tf

from celeba import CelebA
from CondVAE import CVAE
from utils import read_VarToSave, batch_generator, get_parameter



def generate_image_random(model, test_data, name = None, target_attr = None, std=0.01):
    """
    Generate and plot 10 images with a given attribute (if given).
    - list target_attr : list of desidered attributes [default None]
    """
    # Vector of specified attributes
    if target_attr:       
        attr_vect = np.zeros(test_data["n_attr"])
        for attr in target_attr:
            attr_vect[attr] = 1
        name = name + "Attributes"
        print("Generation of 16 images with attributes: ", target_attr )

    # Vector of fixed attributes
    else:        
        #attr_vect = np.random.choice([0, 1], size=(test_data["n_attr"],), p=[2./3, 1./3])
        target_attr = [29, 20, 31, 22, 2, 11, 33, 15]
        attr_vect = np.zeros(test_data["n_attr"])
        for attr in target_attr:
            attr_vect[attr] = 1
        name = name + "_fixedAttr"
        print("Generation of 16 images with fixed attributes: ", target_attr)

    _y = np.tile(attr_vect, reps = [model.batch_size, 1])
    generated = model.decode(label = _y)
    
    #-----------Plot----------------
    imshow_grid(generated, shape=[4, 4], name = name, save = True)


def reconstruct(model, test_data, save_path=None):
    """
    Reconstruct and plot 5 input images.
    """
    batch_gen = batch_generator(test_data['batch_dim'], test_data['test_labels'])
    _x, _y = next(batch_gen)
    reconstruction = model.reconstruct(_x, _y)

    #-----------Plot----------------
    plt.figure(figsize=(6, 10))

    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(_x[i].reshape(64, 64, 3))
        plt.title("Test input %i" % np.argmax(i))
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(reconstruction[i].reshape(64, 64, 3))
        plt.title("Reconstruction")
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "reconstruction.png", bbox_inches="tight")
        plt.clf()

    print("Reconstruction of 5 images from the test set.")

def imshow_grid(imgs, shape=[2, 5], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
    size = shape[0] * shape[1]

    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(imgs[i].reshape(64, 64, 3))  
    if save:
        plt.savefig(str(name) + '.png')
        plt.clf()
    else:
        plt.show()

if __name__ == '__main__':

    #----------------Parser------------------

    parser = argparse.ArgumentParser(description='This is a plot script for the CondVAE.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--plot_type', action='store', nargs='?', const=None, default=None, type=str,
                        choices=None, metavar=None, help="""Plot type.\n- reconstr: reconstruction - gen: generate 10 images with random/given attributes. [default None (all)] """)
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Latent dimension. [default: 20]', metavar=None)
    parser.add_argument('-p', '--progress', action='store', nargs='?', const=None, default=None, type=str, metavar=None,
                        choices=None, help='Use progress model (model is saved each 50 epoch). [default: None]')
    parser.add_argument('-s', '--std', action='store', nargs='?', const=None, default=0.1, type=float,
                        choices=None, help='Std of gaussian noise for `gen_rand` plotting. [default: 0.1]', metavar=None)
    parser.add_argument('-t', '--target', action='store', nargs='?', const=None, default=False, type=bool, metavar=None,
                        choices=None,
                        help='Target attribute(s). [default: False (random attributes)]')
    args = parser.parse_args()

    
    pr = "progress-%s-" % args.progress if args.progress else ""
    param = get_parameter("./parameters.json", args.z_dim)

    #--------- Read test_data.pickle ------------
    test_data = read_VarToSave("./test_data")

    #----------Set the model------------------
    acc = np.load("./log/CVAE_%i/%sacc.npz" % (args.z_dim, pr))
    opt = dict(nn_architecture=param, batch_size=acc["batch_size"])
    opt["label_dim"] = test_data["n_attr"]
    opt["dropout"] = 0
    model = CVAE(load_model="./log/CVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **opt)
    
    # Save path
    folder = "./results/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    #-------------Plot-----------

    # Reconstruction
    if args.plot_type is None or args.plot_type == "reconstr":
        reconstruct(model, test_data, save_path=folder) 
    
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

        generate_image_random(model, test_data, name = folder + "random_generation", target_attr = target_attr, std=args.std)


