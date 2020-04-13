import argparse
import os
import sys
import json

from celeba import CelebA
from CondVAE import CVAE 
from utils import get_parameter, save_VarToSave


if __name__ == '__main__':
    #------Ignore warning message by tensor flow--------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #----------------Parser------------------
    parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Dimension of latent vector. [default: 20]', metavar=None)
    parser.add_argument('-bs', '--batch_size', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Batch size. [default: 100]', metavar=None)
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=150, type=int,
                        choices=None, help='Epoch number. [default: 150]', metavar=None)
    parser.add_argument('-b', '--beta', action='store', nargs='?', const=None, default=1, type=float,
                        choices=None, help='Beta parameter. [default: 1]', metavar=None)
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.005, type=float,
                        choices=None, help='Learning rate. [default: 0.005]', metavar=None)
    parser.add_argument('-td', '--train_dim', action='store', nargs='?', const=None, default=0.8, type=float,
                        choices=None, help=' training set dimension wrt the whole dataset. [default: 0.8]', metavar=None)
    parser.add_argument('-d', '--dropout', action='store', nargs='?', const=None, default=0, type=float,
                        choices=None, help='Dropout parameter. [default: 0]', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=1, type=float,
                        choices=None, help='Gradient clipping. [default: 1]', metavar=None)
    parser.add_argument('-p', '--plot', action='store', nargs='?', const=None, default=False, type=bool,
                        choices=None, help='Plot after train. [default: False]', metavar=None)               
    args = parser.parse_args()

    print("\n Start train the CVAE \n")
    
    #---------------Parameters----------------
    save_path = "./log/CVAE_%i/" % (args.z_dim)
    param = get_parameter("./parameters.json", args.z_dim)
    opt = dict(nn_architecture=param, batch_size=args.batch_size, image_dim = 64*64*3,beta = args.beta, learning_rate=args.lr, save_path=save_path,
               max_grad_norm=args.clip, dropout = args.dropout)

    
    #--------------Prepare Dataset-----------------
    dataset = CelebA(train_dim = args.train_dim)

    #----------Model training on CelebA------------
    opt["label_dim"] = dataset.n_attr    
    model = CVAE(**opt)
    dataset.celebA_train(model, epoch=args.epoch, save_path=save_path)

    #----------Save train and test set information for plot------------
    if args.plot:
        test_data = {
            'train_dim' : dataset.train_dim,
            'n_attr' : dataset.n_attr,
            'test_labels' : dataset.test_labels,
            'attr' : dataset.attr,
            'batch_dim' : args.batch_size
         }

        file_path = "./test_data"
        save_VarToSave(file_path, test_data)

        print("Test labels successfully saved.")