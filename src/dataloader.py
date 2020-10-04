import kaggle
import os


dataset_dir = '/input/CelebA'


# -------------------------------
#  DOWNLOAD DATASET CELEBA
# -------------------------------

def download_celabA(dataset_dir):
    """
    Downloads CelebA dataset from Kaggle and loads it in dataset_dir.
    """
    if not os.path.exists(dataset_dir):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset="jessicali9530/celeba-dataset", path=dataset_dir, unzip=True)
        print('Download completed.')
    else:
        print('CelebA dataset already exists.')
    
    return True


if __name__ == '__main__':

    download_celabA(dataset_dir)
