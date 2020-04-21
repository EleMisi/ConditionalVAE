import argparse
import os
import sys
import json

from celeba import CelebA
from DenseCondVAE import DenseCVAE
from ConvolutionalCondVAE import ConvCVAE 
from utils import get_parameter, save_VarToSave


if __name__ == '__main__':

    #----------------Parser------------------
    parser = argparse.ArgumentParser(description='Conditional VAE train script.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-a', '--alpha', action='store', nargs='?', const=None, default=1, type=float,
                        choices=None, help='Alpha parameter. [default: 1]', metavar=None)
    parser.add_argument('-b', '--beta', action='store', nargs='?', const=None, default=1, type=float,
                        choices=None, help='Beta parameter. [default: 1]', metavar=None)
    parser.add_argument('-bs', '--batch_size', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Batch size. [default: 100]', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=1, type=float,
                        choices=None, help='Gradient clipping. [default: 1]', metavar=None)
    parser.add_argument('-d', '--dropout', action='store', nargs='?', const=None, default=0, type=float,
                        choices=None, help='Dropout parameter. [default: 0]', metavar=None)
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=150, type=int,
                        choices=None, help='Epoch number. [default: 150]', metavar=None) 
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.005, type=float,
                        choices=None, help='Learning rate. [default: 0.005]', metavar=None)
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=32, type=int,
                        choices=None, help='Dimension of latent vector. [default: 32]', metavar=None)
    parser.add_argument('-nn', '--neural_network', action='store', nargs='?', const=None, default='Conv', type=str,
                        choices=None, help='Neural network architecture. [default: Conv]', metavar=None)
    parser.add_argument('-p', '--plot', action='store', nargs='?', const=None, default=False, type=bool,
                        choices=None, help='Plot after train. [default: False]', metavar=None) 
    parser.add_argument('-td', '--train_dim', action='store', nargs='?', const=None, default=0.8, type=float,
                        choices=None, help=' training set dimension wrt the whole dataset. [default: 0.8]', metavar=None)              
    args = parser.parse_args()

    
    # Model constructor parameters
    opt = dict(
        batch_size=args.batch_size, 
        alpha = args.alpha,
        beta = args.beta, 
        learning_rate=args.lr, 
        max_grad_norm=args.clip, 
        dropout = args.dropout)

    
    # Prepare dataset
    dataset = CelebA(train_dim = args.train_dim)

    # Set the network
    if args.neural_network == "Dense":
        param = get_parameter("./parameters.json", args.z_dim) 
        save_path = "./log/DenseCVAE_%i/" % (args.z_dim)
        opt["label_dim"] = dataset.n_attr
        opt["nn_architecture"]=param   
        opt["save_path"] = save_path
        model = DenseCVAE(**opt)
        print("Dense CVAE built.")
        
    if args.neural_network == "Conv":
        save_path = "./log/ConvCVAE_%i/" % (args.z_dim)
        opt["label_dim"] = dataset.n_attr
        opt["latent_dim"] = args.z_dim
        opt["save_path"] = save_path
        model = ConvCVAE(**opt)
        print("Convolutional CVAE built.")


    # Train 
    dataset.celebA_train(model, epoch=args.epoch, save_path=save_path)

    # Save test_data for plot
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