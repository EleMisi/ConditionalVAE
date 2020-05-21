import numpy as np
from keras.applications.inception_v3 import InceptionV3
from scipy import linalg

from metrics_utils import generation_activations, reconstruction_activations

class Metrics:


    @staticmethod
    def compute_metrics(model, n_batches, batch_gen, sample_size, mode = None):

        inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        splits = 10
        reconstr_fid = None
        gen_fid = None
        # Compute metrics for reconstruction
        if mode == None or mode == 'reconstr' :
            print('\nComputing real and reconstructed images activations...\n')
            real_activations_list, fake_activations_list = reconstruction_activations(model, batch_gen, n_batches, inception_model)
            real_activations = np.concatenate(real_activations_list)[:sample_size][:]
            fake_activations = np.concatenate(fake_activations_list)[:sample_size][:]
            print("Real and reconstructed activations dimension: {}, {}".format(real_activations.shape, fake_activations.shape))
            # Compute FID
            reconstr_fid = Metrics.fid(real_activations,fake_activations)

        # Compute metrics for generation
        if mode == None or mode == 'gen':
            print('\nComputing real and generated images activations...\n')
            real_activations_list, fake_activations_list = generation_activations(model, batch_gen, n_batches, inception_model)
            real_activations = np.concatenate(real_activations_list)[:sample_size][:]
            fake_activations = np.concatenate(fake_activations_list)[:sample_size][:]
            print("Real and generated activations dimension: {}, {}".format(real_activations.shape, fake_activations.shape))
            # Compute FID
            gen_fid = Metrics.fid(real_activations,fake_activations)
        

        return reconstr_fid, gen_fid, 


    #---------------------------------
    #   Frechet Inception Distance
    #---------------------------------
        
    @staticmethod
    def fid(real_act, fake_act, eps = 1e-6):
        """
        Returns the Frechet Inception Distance given the activations both for real and fake images.
        """
        print("\nComputing FID...\n")

        # Real and fake images activations mean and covariance
        mu, cov = Metrics.get_statistics(real_act)
        mu_w, cov_w = Metrics.get_statistics(fake_act)

        assert mu.shape == mu_w.shape, "\nmu and mu_w have different dimension: {} and {}\n".format(mu.shape, mu_w.shape)
        assert cov.shape == cov_w.shape, "\nCov and Cov_w have different dimension: {} and {}\n".format(cov.shape, cov_w.shape)
         
        # ||mu - mu_w||^2 
        mean_diff = mu - mu_w
        quadratic_mean_diff = mean_diff.dot(mean_diff)

        # covmean = (Cov * Cov_w)^(1/2)
        # errrest: Frobenius norm of the estimated error, ||err||_F / ||A||_F
        covmean, errest = linalg.sqrtm(cov.dot(cov_w), disp = False)
        #print("\nFrobenius norm of the estimated error: {}".format(errest))

        # Check if there are any NaN or infinite in covmean due to the matrix product and the matrix square root.
        # If this is the case, add a small constant to the covariance matrices to prevent NaN or infinite covmean elements.
        if not np.isfinite(covmean).all():
            warnings.warn("covmean has singular element(s), adding {} to diagonal of cov and cov_w".format(eps))
            diag_eps = np.eye(cov.shape[0]) * eps
            covmean = linalg.sqrtm((cov + diag_eps).dot(cov_w + diag_eps))
            if not np.isfinite(covmean).all():
                raise ValueError("Singular elements not eliminated.")
        
        # Check if there are imaginary numbers in covmean due to numerical error
        if np.iscomplexobj(covmean):
            # Check the magnitude of the imaginary component, if it is greater than 1e-3, than raise an error.
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Maximum imaginary component: {}, with tolerance {} ".format(m, 1e-3))           
            covmean = covmean.real

        # ||mu - mu_w||^2 + Tr(Cov + Cov_w - 2(Cov * Cov_w)^(1/2))
        # Trace(A + B) == Trace(A) + Trace(B)
        fid = quadratic_mean_diff + np.trace(cov) + np.trace(cov_w) - 2 * np.trace(covmean)

        return fid


    @staticmethod
    def get_statistics(act):
        """
        Returns the mean and the covariance of the given array of activations.
        """

        mu = np.mean(act, axis=0)
        cov = np.cov(act, rowvar=False)

        return mu, cov

