import argparse
import os
import sys
import json

from celeba import CelebA
from CondVAE import CVAE 
from utils import celebA_train, get_parameter


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
    save_path = "./log/CVAE_%i/" % (args.z_dim)
    param = get_parameter("./parameters.json", args.z_dim)
    opt = dict(nn_architecture=param, batch_size=args.batch_size, learning_rate=args.lr, save_path=save_path,
               max_grad_norm=args.clip)

    
    #--------------Prepare Dataset-----------------

    dataset = CelebA(train_dim=0.80)

    #----------Model training on CelebA------------
    opt["label_dim"] = dataset.n_attr    
    model = CVAE(**opt)
    celebA_train(model, dataset, epoch=args.epoch, save_path=save_path)

    #----------Store test set------------
    
    #To develop