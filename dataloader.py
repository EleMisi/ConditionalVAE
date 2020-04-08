from collections import OrderedDict
from glob import glob
import kaggle
import numpy as np
import os
import pandas as pd
from PIL import Image
import json

dataset_dir = '/input/CelebA'


# ---------------------------------------------------------------------------------------------------------------------
#  DOWNLOAD DATASET CELEBA
# ---------------------------------------------------------------------------------------------------------------------

def download_celabA(dataset_dir):
    """
    Download CelebA dataset in dataset_dir.
    @BOG:
    Ho eliminato il tuo username e la tua kaggle_key per caricare su github il file senza dati sensibili
    """
    #os.environ['KAGGLE_USERNAME'] = 
    #os.environ['KAGGLE_KEY'] = 
    if not os.path.exists(dataset_dir):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset="jessicali9530/celeba-dataset", path=dataset_dir, unzip=True)
        print('Download completed.')

    else:
        print('CelebA dataset already exists.')

    return True

if __name__ == '__main__':

    download_celabA(dataset_dir)
