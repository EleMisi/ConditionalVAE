import argparse
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf



#####################################
#    Conditional Image Generation   #
#####################################


def image_generation(model, test_data,target_attr = None, save_path = None):
    """
    Generates and plots 16 images with a given attribute (if given).
    - list target_attr : list of desidered attributes [default None]
    """
    # Vector of user-defined attributes.
    if target_attr:       
        attr_vect = np.zeros(len(test_data["attributes"]))
        for attr in target_attr:
            attr_vect[attr] = 1
        labels = np.tile(attr_vect, reps = [test_data['batch_size'], 1])
        print("Generation of 16 images with attributes: ", target_attr )

   # Vector of attributes taken from the test set.
    else:        
        batch_gen = batch_generator(test_data['batch_size'], test_data['test_labels'], model_name = 'Conv')
        _, labels = next(batch_gen)
        print("Generation of 16 images with fixed attributes.")


    z_cond = model.reparametrization(input_label = labels, batch_size = test_data['batch_size'])
    logits = model.decoder(z_cond, is_train = False)
    generated = tf.nn.sigmoid(logits)
  

    # Plot
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(1,1,1)
    ax.imshow(convert_batch_to_image_grid(generated.numpy()))
    plt.axis('off')

    if save_path :
      plt.savefig(save_path + "generation_" + str(target_attr) + "_.png")
    
    plt.show()
    plt.clf()

    
#############################
#    Image Reconstruction   #
#############################

def image_reconstruction(model, test_data, save_path=None):
    """
    Reconstructs and plots 32 test images.
    """
    batch_gen = batch_generator(test_data['batch_size'], test_data['test_labels'], model_name = 'Conv')
    images, labels= next(batch_gen)
    model_output= model((images, labels), is_train = False)
    
    f = plt.figure(figsize=(64,40))
    ax = f.add_subplot(1,2,1)
    ax.imshow(convert_batch_to_image_grid(images),
              interpolation='nearest')
    plt.axis('off')

    ax = f.add_subplot(1,2,2)
    ax.imshow(convert_batch_to_image_grid(model_output['recon_img'].numpy()) ,
              interpolation='nearest')
    plt.axis('off')
    
    if save_path :
      plt.savefig(save_path + "reconstruction.png")

    plt.show()
    plt.clf()

    print("Reconstruction of a batch of test set images.")
    
    
 import cv2



#############
#    Utils  #
#############

def batch_generator(batch_dim, test_labels, model_name):
    """
    Batch generator using the given list of labels.
    """
    while True:
        batch_imgs = []
        labels = []
        for label in (test_labels):
            labels.append(label)
            if len(labels) == batch_dim:
                batch_imgs = create_image_batch(labels, model_name)
                batch_labels = [x[1] for x in labels]
                yield np.asarray(batch_imgs), np.asarray(batch_labels)
                batch_imgs = []
                labels = []
                batch_labels = []
        if batch_imgs:
            yield np.asarray(batch_imgs), np.asarray(batch_labels)


def get_image(image_path, model_name, img_size = 128, img_resize = 64, x = 25, y = 45):
    """
    Crops, resizes and normalizes the target image.
        - If model_name == Dense, the image is returned as a flattened numpy array with dim (64*64*3)
        - otherwise, the image is returned as a numpy array with dim (64,64,3)
    """

    img = cv2.imread(image_path)
    img = img[y:y+img_size, x:x+img_size]
    img = cv2.resize(img, (img_resize, img_resize))
    img = np.array(img, dtype='float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img /= 255.0 # Normalization to [0.,1.]

    if model_name == "Dense" :
        img = img.ravel()
    
    return img


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


def convert_batch_to_image_grid(image_batch, dim = 64):
    reshaped = (image_batch.reshape(4, 8, dim, dim, 3)
              .transpose(0, 2, 1, 3, 4)
              .reshape(4 * dim, 8 * dim, 3))
    return reshaped 
