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
from utils import get_parameter, next_batch



def generate_image_random(model, dataset, name = None, target_attr = None, std=0.01):
    """
    Generate and plot 10 images with a given attribute (if given).
    - list target_attr : list of desidered attributes [default None]
    """
    #Vector of specified attributes
    if target_attr:       
        attr_vect = np.zeros(dataset["n_attr"])
        for attr in target_attr:
            attr_vect[dataset["attr"][attr]] = 1
        name = name + "Attr"
        print("Generation of 10 images with attributes: ", target_attr )

    #Vector of random attributes
    else:        
        attr_vect = np.random.choice([0, 1], size=(dataset["n_attr"],), p=[2./3, 1./3])
        name = name + "_noAttr"
        print("Generation of 10 images with random attributes.")

    true_label = np.tile(attr_vect, reps = [model.batch_size, 1])
    generated = model.decode(true_label, std = std)
    
    #-----------Plot----------------
    imshow_grid(generated, shape=[2, 5], name = name, save = True)


def reconstruct(model, dataset, save_path=None):
    """
    Reconstruct and plot 5 input images.
    """
    _x, _y = next_batch(model.batch_size, dataset["test_set"], dataset["test_labels"])
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

    # Read test_data 
    with open('test_data.json', 'r') as openfile: 
        dataset = json.load(openfile) 

    # Set the model
    acc = np.load("./log/CVAE_%i/%sacc.npz" % (args.z_dim, pr))
    opt = dict(nn_architecture=param, batch_size=acc["batch_size"])
    opt["label_dim"] = dataset["n_attr"] 
    model = CVAE(load_model="./log/CVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **opt)
    
    # Save path
    folder = "./results/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    #-------------Plot-----------

    # Reconstruction
    if args.plot_type is None or args.plot_type == "reconstr":
        reconstruct(model, dataset, save_path=folder) 
    
    # Generation 
    if args.plot_type is None or args.plot_type == "gen":
        target_attr = None
        if args.target :
            try:
                n = input("Number of desired attributes?", "(type help for instructions)", sep = '\n')
                if n == "help":
                    print("The attributes must insert as a list of integers: 1 3 4 5 3","Here is encoding:", sep = '\n')
                    for k, v in dataset['attr'].items():
                        print(k, "-->", v)
                    n = int(input("Number of desired attributes?"))
                else:
                    n = int(n)

                target_attr = list(map(int,input("\nEnter the desired attributes: ").strip().split()))[:n] 

            except TypeError:
                print("Invalid input!")
        
        generate_image_random(model, dataset, name = folder + "random_generation", target_attr = target_attr, std=args.std)


