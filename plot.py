import sys
import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import load_data, next_batch, prepare_dataset, data_and_labels


def get_parameter(path, z_dim):
    with open(path) as f:
        p = json.load(f)
    if z_dim:
        p["z_dim"] = z_dim
    return p


def generate_image_random(model, feeder, save_path=None, target_attr=0, std=0.01, input_image=False, seed=False):
    """
    Generate and plot 10 images with a given attribute.
    If seed = True the images are generated starting from a latent vector.
    Otherwise the generation starts from a vector drawn from prior in latent space.
    """

    #----------Image generation with seed------------
    if seed:
        # Latent vector generation
        if target_attr == 1:
            _code = []
            for i in range(10):
                _x, _y = next_batch(model.batch_size, feeder[0],feeder[1])
                __code = model.encode(_x, _y).tolist()
                for __x, __y, _c in zip(_x, _y, __code):
                    if np.argmax(__y) == target_attr:
                        _code.append(_c)
            
            # Binary vector of attributes
            attr_vect = np.zeros(model.label_dim)
            attr_vect[target_attr] = 1
            _len = model.batch_size
            z = np.tile(np.mean(_code, axis = 0), [_len, 1])

        # Image generation
        true_label = np.tile(attr_vect, reps=[_len, 1])
        z += np.random.randn(model.batch_size, model.nn_architecture["z_dim"]) * std
        generated = model.decode(true_label, z)

    #----------Image generation without seed------------

    else:
        if target_attr:
            attr_vect = np.zeros(model.label_dim)
            attr_vect[target_attr] = 1
            _len = model.batch_size
        else:
            target_vector = [i for i in range(model.label_dim)]
            attr_vect = np.eye(model.label_dim)[target_vector]
            _len = int(model.batch_size / model.label_dim)

        true_label = np.tile(attr_vect, reps=[_len, 1])
        generated = model.decode(true_label, std=std)
    
    #-----------Plot----------------

    plt.figure(figsize=(6, 10))
    n_images = 10
    for i in range(n_images):
        plt.subplot(5, 2, i + 1)
        plt.imshow(generated[i].reshape(64, 64, 3))
        plt.title("Generated %i" % np.argmax(true_label[i]))
        plt.tight_layout()
    if save_path:
        plt.savefig("%sRandom_images_%s_%0.3f.png" % (save_path, str(target_attr), std), bbox_inches="tight")


def generate_image_mean(model, feeder, save_path=None, input_image=False):
    """
    Generate and plot 10 images by the mean latent vector.
    """
    # Latent vector generation
    _code = {x : [] for x in range(model.label_dim)}

    for i in range(10):
        _x, _y = next_batch(model.batch_size, feeder[0], feeder[1])
        __code = model.encode(_x, _y).tolist()
        for __x, __y, _c in zip(_x, _y, __code):
            _code[int(np.argmax(__y))].append(_c)

    _temp = np.vstack([np.mean(_code[_a], 0) for _a in range(model.label_dim)])
    z = np.tile(_temp, [int(model.batch_size / model.label_dim), 1])

    # Binary vector of attributes
    attr_vect = np.eye(model.label_dim)[[i for i in range(model.label_dim)]]
    true_label = np.tile(attr_vect, [int(model.batch_size / model.label_dim), 1])

    # Image generation
    generated = model.decode(true_label, z)

    #-----------Plot----------------

    plt.figure(figsize=(6, 10))
    n_images = 10
    for i in range(n_images):
        plt.subplot(5, 2, i + 1)
        plt.imshow(generated[i].reshape(64, 64, 3))
        plt.title("Generated %i" % np.argmax(true_label[i]))
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "Generated_image_mean.png", bbox_inches="tight")


def reconstruct(model, feeder, n_images=5, save_path=None, input_image=False):
    """
    Reconstruct and plot 5 input images.
    """
    _x, _y = next_batch(model.batch_size, feeder[0], feeder[1])
    reconstruction = model.reconstruct(_x, _y)

    #-----------Plot----------------
    plt.figure(figsize=(8, 12))

    for i in range(n_images):
        plt.subplot(n_images, 2, 2 * i + 1)
        plt.imshow(_x[i].reshape(64, 64, 3))
        plt.title("Test input %i" % np.argmax(_y[i]))
        plt.colorbar()
        plt.subplot(n_images, 2, 2 * i + 2)
        plt.imshow(reconstruction[i].reshape(64, 64, 3))
        plt.title("Reconstruction")
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "Reconstruction.png", bbox_inches="tight")


if __name__ == '__main__':

    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)

    #----------------Parser------------------

    parser = argparse.ArgumentParser(description='This is a plot script for the CondVAE.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--plot_type', action='store', nargs='?', const=None, default=None, type=str,
                        choices=None, metavar=None, help="""Plot type.\n- reconstr: reconstruction - gen_ave: generate image by the mean latent vector - gen_rand: generate image by the mean latent vector with additive gaussian noise. - embed: embed 2 dimension for visualization""")
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Latent dimension. [default: 20]', metavar=None)
    parser.add_argument('-p', '--progress', action='store', nargs='?', const=None, default=None, type=str, metavar=None,
                        choices=None, help='Use progress model (model is saved each 50 epoch). [default: None]')
    parser.add_argument('-s', '--std', action='store', nargs='?', const=None, default=0.1, type=float,
                        choices=None, help='Std of gaussian noise for `gen_rand` plotting. [default: 0.1]', metavar=None)
    parser.add_argument('-t', '--target', action='store', nargs='?', const=None, default=None, type=int, metavar=None,
                        choices=None,
                        help='Target attribute(s) to generate for `gen_rand` plotting. [default: None (all attributes)]')
    parser.add_argument('-sd', '--seed', action='store', nargs='?', const=None, default=False, type=bool, metavar=None,
                        choices=None,
                        help='If use seed for `gen_rand` plotting. [default: True]')
    args = parser.parse_args()

    
    print("\n Plot training results ...")

    
    pr = "progress-%s-" % args.progress if args.progress else ""
    param = get_parameter("./parameters.json", args.z_dim)

    #-----------Set the model-------------

    acc = np.load("./log/CVAE_%i/%sacc.npz" % (args.z_dim, pr))
    opt = dict(nn_architecture=param, batch_size=acc["batch_size"])
    opt["label_dim"] = 2 #Provvisorial small dataset
    from CondVAE import CVAE
    _inp_img = False
    model = CVAE(load_model="./log/CVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **opt)
    dataset_dict = prepare_dataset()
    n_samples = len(dataset_dict)
    imgs, labels = data_and_labels(dataset_dict)
    n_batches = int(n_samples / model.batch_size)
    celeba = (imgs, labels)

    fig_path = "./results/ProvvisorialDataSet-" 
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    #-------------Plot-----------
    
    # Reconstruction
    if args.plot_type is None or args.plot_type == "reconstr":
        reconstruct(model, feeder = celeba, input_image=_inp_img, save_path=fig_path) 
    # Generation by mean 
    if args.plot_type is None or args.plot_type == "gen_ave":
        generate_image_mean(model, feeder = celeba, input_image=_inp_img, save_path=fig_path)
    # Random generation
    if args.plot_type is None or args.plot_type == "gen_rand":
        generate_image_random(model, feeder = celeba, input_image=_inp_img, save_path=fig_path, std=args.std, seed=args.seed)



