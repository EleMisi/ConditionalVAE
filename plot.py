import sys
import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import mnist_loader, shape_2d


def get_parameter(path, latent_dim):
    with open(path) as f:
        p = json.load(f)
    if latent_dim:
        p["dim_z"] = latent_dim
    return p


def plot_2d_embedded(model, feeder, mode, save_path=None, input_image=False, n=10000):
    y, z, cnt = [], [], 0
    while cnt < n:
        _x, _y = feeder.train.next_batch(model.batch_size)
        y.append(_y)
        _x = shape_2d(_x, model.batch_size) if input_image else _x
        if mode == "conditional":
            _z = model.encode(_x, _y)
        elif mode == "unsupervised":
            _z = model.encode(_x)
        else:
            sys.exit("unknown mode %s !" % mode)
        z.append(_z)
        cnt += model.batch_size
    z = np.vstack(z)
    y = np.vstack(y)
    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(y, 1))
    plt.colorbar()
    plt.grid()
    if save_path:
        # plt.savefig("%sembedding.eps" % save_path, bbox_inches="tight")
        plt.savefig("%sembedding.png" % save_path, bbox_inches="tight")


def generate_image_random(model, feeder, save_path=None, n=10, target_digit=None, std=0.01, input_image=False,
                          seed=True):
    # generate latent vector
    if seed:
        if target_digit:
            _code = []
            for i in range(500):
                _x, _y = feeder.test.next_batch(model.batch_size)
                _x = shape_2d(_x, model.batch_size) if input_image else _x
                __code = model.encode(_x, _y).tolist()
                for __x, __y, _c in zip(_x, _y, __code):
                    if np.argmax(__y) == target_digit:
                        _code.append(_c)
            o_h = np.zeros(model.label_dim)
            o_h[target_digit] = 1
            _len = model.batch_size
            z = np.tile(np.mean(_code, 0), [_len, 1])
        else:
            target_digit = "all"
            _code = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
            for i in range(500):
                _x, _y = mnist.test.next_batch(model.batch_size)
                _x = shape_2d(_x, model.batch_size) if input_image else _x
                __code = model.encode(_x, _y).tolist()
                for __x, __y, _c in zip(_x, _y, __code):
                    _code[int(np.argmax(__y))].append(_c)
            # convert label to one hot vector
            target_vector = [i for i in range(model.label_dim)]
            o_h = np.eye(model.label_dim)[target_vector]
            _len = int(model.batch_size / model.label_dim)
            tmp = np.vstack([np.mean(_code[_a], 0) for _a in range(model.label_dim)])
            z = np.tile(tmp, [_len, 1])
        true_label = np.tile(o_h, reps=[_len, 1])
        z += np.random.randn(model.batch_size, model.nn_architecture["z_dim"]) * std
        generated = model.decode(true_label, z)
    else:
        if target_digit:
            o_h = np.zeros(model.label_dim)
            o_h[target_digit] = 1
            _len = model.batch_size
        else:
            target_vector = [i for i in range(model.label_dim)]
            o_h = np.eye(model.label_dim)[target_vector]
            _len = int(model.batch_size / model.label_dim)
        true_label = np.tile(o_h, reps=[_len, 1])
        generated = model.decode(true_label, std=std)

    plt.figure(figsize=(6, 10))
    for i in range(n):
        plt.subplot(5, 2, i + 1)
        plt.imshow(generated[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Generated %i" % np.argmax(true_label[i]))
        plt.colorbar()
        plt.tight_layout()
    if save_path:
        # plt.savefig("%sgenerated_image_rand_%i_%0.3f.eps" % (save_path, target_digit, std), bbox_inches="tight")
        plt.savefig("%sgenerated_image_rand_%s_%0.3f.png" % (save_path, str(target_digit), std), bbox_inches="tight")


def generate_image_mean(model, feeder, save_path=None, input_image=False):
    _code = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for i in range(500):
        _x, _y = feeder.test.next_batch(model.batch_size)
        _x = shape_2d(_x, model.batch_size) if input_image else _x
        __code = model.encode(_x, _y).tolist()
        for __x, __y, _c in zip(_x, _y, __code):
            _code[int(np.argmax(__y))].append(_c)

    # convert label to one hot vector
    o_h = np.eye(model.label_dim)[[i for i in range(model.label_dim)]]
    true_label = np.tile(o_h, [int(model.batch_size / model.label_dim), 1])

    tmp = np.vstack([np.mean(_code[_a], 0) for _a in range(model.label_dim)])
    z = np.tile(tmp, [int(model.batch_size / model.label_dim), 1])

    generated = model.decode(true_label, z)
    plt.figure(figsize=(6, 10))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.imshow(generated[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Generated %i" % np.argmax(true_label[i]))
        plt.colorbar()
        plt.tight_layout()
    if save_path:
        # plt.savefig(save_path + "generated_image_mean.eps", bbox_inches="tight")
        plt.savefig(save_path + "generated_image_mean.png", bbox_inches="tight")


def plot_reconstruct(model, mode, feeder, _n=5, save_path=None, input_image=False):
    # feed test data and reconstruct
    _x, _y = feeder.test.next_batch(model.batch_size)
    _x = shape_2d(_x, model.batch_size) if input_image else _x
    if mode == "conditional":
        reconstruction = model.reconstruct(_x, _y)
    elif mode == "unsupervised":
        reconstruction = model.reconstruct(_x)
    # plot
    plt.figure(figsize=(8, 12))
    for i in range(_n):
        plt.subplot(_n, 2, 2 * i + 1)
        plt.imshow(_x[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input %i" % np.argmax(_y[i]))
        plt.colorbar()
        plt.subplot(_n, 2, 2 * i + 2)
        plt.imshow(reconstruction[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
        plt.tight_layout()
    if save_path:
        # plt.savefig(save_path + "reconstruction.eps", bbox_inches="tight")
        plt.savefig(save_path + "reconstruction.png", bbox_inches="tight")


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # set NumPy random seed
    np.random.seed(10)
    # set TensorFlow random seed
    tf.set_random_seed(10)
    # Parser
    parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--plot_type', action='store', nargs='?', const=None, default=None, type=str,
                        choices=None, metavar=None, help="""Plot type.\n- reconstr: reconstruction
- gen_ave: generate image by the mean latent vector
- gen_rand: generate image by the mean latent vector with additive gaussian noise.
- embed: embed 2 dimension for visualization""")
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Latent dimension. [default: 20]', metavar=None)
    parser.add_argument('-p', '--progress', action='store', nargs='?', const=None, default=None, type=str, metavar=None,
                        choices=None, help='Use progress model (model is saved each 50 epoch). [default: None]')
    parser.add_argument('-s', '--std', action='store', nargs='?', const=None, default=0.1, type=float,
                        choices=None, help='Std of gaussian noise for `gen_rand` plotting. [default: 0.1]', metavar=None)
    parser.add_argument('-t', '--target', action='store', nargs='?', const=None, default=None, type=int, metavar=None,
                        choices=None,
                        help='Target digit to generate for `gen_rand` plotting. [default: None (plot all digit)]')
    parser.add_argument('-sd', '--seed', action='store', nargs='?', const=None, default=True, type=bool, metavar=None,
                        choices=None,
                        help='If use seed for `gen_rand` plotting. [default: True]')
    args = parser.parse_args()

    print("\n Plot the result of training")
    pr = "progress-%s-" % args.progress if args.progress else ""
    param = get_parameter("./parameters.json", args.z_dim)
    acc = np.load("./log/CVAE_%i/%sacc.npz" % (args.z_dim, pr))
    opt = dict(nn_architecture=param, batch_size=acc["batch_size"])

    from CondVAE import CVAE as Model
    _mode, _inp_img = "conditional", False

    if _mode == "conditional":
        opt["label_dim"] = 10

    model_instance = Model(load_model="./log/CVAE_%i/%smodel.ckpt" % (args.z_dim, pr), **opt)
    mnist, size = mnist_loader()

    fig_path = "./figure/CVAE_%i/" % (args.z_dim)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    if args.plot_type is None or args.plot_type == "constr":
        plot_reconstruct(model_instance, feeder=mnist, mode=_mode, input_image=_inp_img, save_path=fig_path)
    if args.z_dim == 2:
        if args.plot_type is None or args.plot_type == "embed":
            plot_2d_embedded(model_instance, mnist, mode=_mode, input_image=_inp_img, save_path=fig_path)
    if _mode == "conditional":
        if args.plot_type is None or args.plot_type == "gen_ave":
            generate_image_mean(model_instance, mnist, input_image=_inp_img, save_path=fig_path)
        if args.plot_type is None or args.plot_type == "gen_rand":
            generate_image_random(model_instance, mnist, input_image=_inp_img, save_path=fig_path, std=args.std,
                                  target_digit=args.target, seed=args.seed)



