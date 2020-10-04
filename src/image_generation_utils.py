import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import batch_generator, convert_batch_to_image_grid



#################################
#  Conditional Image Generation #
#################################

def image_generation(model, test_data, target_attr = None, save_path = None):
    """
    Generates and plots a batch of images with specific attributes (if given).
    
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
        batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name = 'Conv')
        _, labels = next(batch_gen)
        print("Generation of 16 images with fixed attributes.")


    z_cond = model.reparametrization(input_label = labels, z_mean = 1.0, z_log_var=0.3)
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


#########################
#  Image Reconstruction #
#########################


def image_reconstruction(model, test_data, save_path=None):
    """
    Reconstructs and plots a bacth of test images.
    """

    batch_gen = batch_generator(test_data['batch_size'], test_data['test_img_ids'], model_name = 'Conv')
    images, labels= next(batch_gen)
    model_output= model((images, labels), is_train = False)
    
    f = plt.figure(figsize=(64,40))
    ax = f.add_subplot(1,2,1)
    ax.imshow(convert_batch_to_image_grid(images))
    plt.axis('off')

    ax = f.add_subplot(1,2,2)
    ax.imshow(convert_batch_to_image_grid(model_output['recon_img'].numpy()))
    plt.axis('off')
    
    if save_path :
      plt.savefig(save_path + "reconstruction.png")

    plt.show()
    plt.clf()

    print("Reconstruction of a batch of test set images.")



############################
#    Image Interpolation   #
############################


def interpolation(target_images, imgs, labels, model):
    """Performs a vector interpolation in the latent space to generate new images."""

    z_vectors = []
    resized_labels = []
    images = []

    # Computing the mean latent vector associated to each image
    for i in target_images:

      img = imgs[i][np.newaxis, ...]
      label = labels[i][np.newaxis, ...]
      model_output = model((img, label), is_train = False)
      img_z = model_output['z_mean']
      #img_var = model_output['z_log_var']
      z_vectors.append(img_z)
      resized_labels.append(label)


    for i in range(4):
      ratios = np.linspace(0, 1, num=8)
      vectors = []

      # Vectors interpolation
      for ratio in ratios:
        v = (1.0 - ratio) * z_vectors[i] + ratio * z_vectors[i+1]
        vectors.append(v)

      vectors = np.asarray(vectors)

      # Generation
      for j,v in enumerate(vectors):
        if j < 4 :
          z_cond = tf.concat([v, resized_labels[i]], axis=1)
          logits = model.decoder(z_cond, is_train = False)
          generated = tf.nn.sigmoid(logits)

        else :
          z_cond = tf.concat([v, resized_labels[i+1]], axis=1)
          logits = model.decoder(z_cond, is_train = False)
          generated = tf.nn.sigmoid(logits)

        images.append(generated.numpy()[0,:,:,:])

    return images


###############################
#   Attributes Manipulation   #
###############################


def attr_manipulation(images, labels, target_attr, model):
    """ Reconstructs a batch of images with modified attributes (target_attr)."""

    reconstructed_images = []
    modified_images = []

    for i in range(images.shape[0]):

        img = images[i][np.newaxis,...]
        label = labels[i][np.newaxis,...]
        model_output = model((img, label), is_train = False)
        img_z = model_output['z_mean']

        reconstructed_images.append(model_output['recon_img'].numpy()[0,:,:,:])

        modified_label = labels[i]
        
        for attr, value in target_attr.items():

            modified_label[attr] = value
            modified_label = modified_label[np.newaxis,...]

        z_cond = tf.concat([img_z, modified_label], axis=1)
        logits = model.decoder(z_cond, is_train = False)
        generated = tf.nn.sigmoid(logits)

        modified_images.append(generated.numpy()[0,:,:,:])

    return np.asarray(reconstructed_images, dtype = 'float32'), np.asarray(modified_images, dtype = 'float32')