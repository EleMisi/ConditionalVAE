import argparse
import os
import sys
import json
from utils import celebA_train, mnist_train

def get_parameter(path, z_dim):
    with open(path) as f:
        p = json.load(f)
    if z_dim:
        p["z_dim"] = z_dim
    return p


if __name__ == '__main__':
    #------Ignore warning message by tensor flow--------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #----------------Parser------------------
    parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Dimension of latent vector. [default: 20]', metavar=None)
    parser.add_argument('-b', '--batch_size', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Batch size. [default: 100]', metavar=None)
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=150, type=int,
                        choices=None, help='Epoch number. [default: 150]', metavar=None)
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.005, type=float,
                        choices=None, help='Learning rate. [default: 0.005]', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=None, type=float,
                        choices=None, help='Gradient clipping. [default: None]', metavar=None)
    args = parser.parse_args()

    print("\n Start train the CVAE \n")
    
    #---------------Parameters----------------
    save_path = "./log/CVAE_%s/" % (args.z_dim)
    param = get_parameter("./parameters.json", args.z_dim)
    opt = dict(nn_architecture=param, batch_size=args.batch_size, learning_rate=args.lr, save_path=save_path,
               max_grad_norm=args.clip)

    #----------Model------------
    from CondVAE import CVAE as Model
    _mode, _inp_img = "conditional", False
    opt["label_dim"] = 10
    print(Model.__doc__)
    model = Model(**opt)
    mnist_train(model=model, epoch=args.epoch, save_path=save_path, mode=_mode, input_image=_inp_img)
