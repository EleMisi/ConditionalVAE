from glob import glob
import json
import logging
import numpy as np
import os
import pickle
from PIL import Image
import sys
import tensorflow as tf


#-----------------------------------------
#   Utils to save and read pickle files
#-----------------------------------------

def save_VarToSave(file_name, varToSave):
    """
    Save varToSave on file_name.pickle
    """
    with open((file_name+'.pickle'), 'wb') as openfile:
        print(type(varToSave))
        pickle.dump(varToSave, openfile)


def read_VarToSave(file_name):
    """
    Read file_name.pickle and return its content
    """
    with (open((file_name+'.pickle'), "rb")) as openfile:
        while True:
            try:
                objects=pickle.load(openfile)
            except EOFError:
                break
    return objects


#-----------------------------------------------
#       Utils for image_reconstruction (plot.py)
#-----------------------------------------------


def batch_generator(batch_dim, test_labels, model_name):
    """
    Batch generator using test set labels.
    """
    while True:
        batch_imgs = []
        labels = []
        for label in (test_labels):
            labels.append(label)
            if len(labels) == batch_dim:
                batch_imgs = create_image_batch(labels, model_name)
                batch_labels = [x[1] for x in labels]
                yield (np.asarray(batch_imgs), np.asarray(batch_labels))
                batch_imgs = []
                labels = []
                batch_labels = []
        if batch_imgs:
            yield (np.asarray(batch_imgs), np.asarray(batch_labels))

def get_image( image_path, model_name, img_size = 128, img_resize = 64, x = 25, y = 45):
    """
    Crops, resizes and normalizes the target image.
    If model_name == Dense, the image is returned as a flattened numpy array with dim (64*64*3)
    Otherwise, the image is returned as a numpy array with dim (64,64,3)
    """
    mode='RGB' 
    image = Image.open(image_path)
    # Crop 
    image = image.crop([x, y, x + img_size, y+img_size])
    # Resize
    image = image.resize([img_resize, img_resize], Image.BILINEAR)
    # Normalization
    img = np.array(image.convert(mode)).astype('float32')
    img /= 255.
    
    if model_name == "Dense" :
        img = img.ravel()

    return np.array(img)


def create_image_batch(labels, model_name):
    """
    Returns the list of images corresponding to the given labels.
    """
    imgs = []
    imgs_id = [item[0] for item in labels]

    for i in imgs_id:
        image_path ='/input/CelebA/img_align_celeba/img_align_celeba/' + i
        imgs.append(get_image(image_path, model_name))

    return imgs



#----------------------------------
#   parameters.json file reader 
#----------------------------------

def get_parameter(path, z_dim):
    with open(path) as f:
        p = json.load(f)
    if z_dim:
        p["z_dim"] = z_dim
    return p



#----------------------------------
#       Utils for celeba_train
#----------------------------------

def create_log(name):
    """Log file creator."""
    if os.path.exists(name):
        os.remove(name)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    # handler for log file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter("H_1, %(asctime)s %(levelname)8s %(message)s"))

    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter("H_2, %(asctime)s %(levelname)8s %(message)s"))
    log.addHandler(handler1)
    log.addHandler(handler2)
    return log


