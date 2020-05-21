import argparse
import time

from metrics import Metrics
from metrics_utils import load_model



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Metrics script for CVAE.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--mode', action='store', nargs='?', const=None, default=None, type=str,
                        choices=None, metavar=None, help='Mode.\n- reconstr: reconstruction - gen: generation. [default None (both)]')
    parser.add_argument('-n', '--z_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='Latent dimension. [default: 20]', metavar=None)
    parser.add_argument('-nn', '--neural_network', action='store', nargs='?', const=None, default='Conv', type=str,
                        choices=None, help='Neural network architecture. [default: Conv]', metavar=None)
    parser.add_argument('-p', '--progress', action='store', nargs='?', const=None, default=None, type=str, metavar=None,
                        choices=None, help='Use progress model. [default: None]')
    parser.add_argument('-s', '--sample_size', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Sample size fot FID computation. [default: 100]', metavar=None) 
    args = parser.parse_args()

    
    # Load the NN model, define the number of batches needed to get the required sample size and create the batch generator.
    model, n_batches, batch_gen = load_model(args)

    # Compute FID e ICP
    start_time = time.time()
    reconstr_fid, gen_fid  = Metrics.compute_metrics(model, n_batches, batch_gen, args.sample_size, args.mode)
    print('\nTotal computation time: {} s\n'.format(int(time.time() - start_time)))
    
    print("-"*80)
    print('Reconstruction FID:', reconstr_fid)  
    print("-"*80)
    print('Generation FID:', gen_fid)
    print("-"*80)