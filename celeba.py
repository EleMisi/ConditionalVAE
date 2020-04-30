from collections import OrderedDict
from glob import glob
import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import sys
import tensorflow as tf

from utils import create_log


class CelebA():


    def __init__(self, train_dim):
        
        self.train_dim = train_dim      
        self.labels, self.attr = self.load_labels()
        self.train_labels, self.test_labels = self.split() 
        self.n_train = len(self.train_labels)
        self.n_attr = len(self.attr)
        

    def get_image(self, image_path, model_name, img_size = 128, img_resize = 64, x = 25, y = 45):
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


    def create_image_batch(self, labels, model_name):
        """
        Returns the list of images corresponding to the given labels.
        """
        imgs = []
        imgs_id = [item[0] for item in labels]

        for i in imgs_id:
            image_path ='/input/CelebA/img_align_celeba/img_align_celeba/' + i
            imgs.append(self.get_image(image_path, model_name))

        return imgs

    def load_labels(self):
        """
        Returns
            labels : OrderedDict
                Key = image id
                Value = binary vector of attributes
            attributes : List
                list of attributes names
        """
        print("\nLoading labels and attributes...\n")

        file_path = "/input/CelebA/list_attr_celeba.csv"
        df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)

        attributes = [x for x in df.columns] 
        od = OrderedDict(df.to_dict('index'))
        labels = OrderedDict()
        for k,v in od.items():
          label = [x for x in v.values()]
          labels[k] = label

        print("Labels: {} \nAttributes: {} \n".format(len(labels), len(attributes)))

        return labels, attributes


    def split(self):
        """
        Returns train set and test set labels as lists of tuples [(im_id1, label_1), ..., (im_idN, label_N)]
        """
        print("\nSplitting dataset...\n")

        # Shuffle 
        shuffled_labels = self.shuffle()
        # Split (according to train_dim)
        n_train = int(len(self.labels) * self.train_dim)
        list_items = list(shuffled_labels.items())
        train_labels = list_items[:n_train]
        test_labels = list_items[n_train:]

        print("Train set dimension: {} \nTest set dimension: {} \n".format(len(train_labels), len(test_labels)))
        
        return train_labels, test_labels
    

    def shuffle(self):
        """
        Returns a shuffled OrderedDict of labels
        """
        items = list(self.labels.items())
        random.shuffle(items)
        shuffled_labels = OrderedDict(items)
    
        return shuffled_labels


    def batch_generator(self, batch_dim, model_name):
        """
        Batch generator using train set labels.
        """
        while True:
            batch_imgs = []
            labels = []
            for label in (self.train_labels):
                labels.append(label)
                if len(labels) == batch_dim:
                    batch_imgs = self.create_image_batch(labels, model_name)
                    batch_labels = [x[1] for x in labels]
                    yield (np.asarray(batch_imgs), np.asarray(batch_labels))
                    batch_imgs = []
                    labels = []
                    batch_labels = []
            if batch_imgs:
                yield (np.asarray(batch_imgs), np.asarray(batch_labels))


    #-------------------------
    #      CelebA Train
    #-------------------------

    def celebA_train(self, model, n_epochs, save_path="./"):
        """
        Training function for CelebA dataset.
        """
        n_batches = int(self.n_train / model.batch_size)

        # Log
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        log = create_log(save_path+"Log")
        log.info("Training set dimension: {}, # of batches: {}, batch dimension {}".format(self.n_train, n_batches, model.batch_size))
        
        # Session initialization
        model.sess.run(tf.compat.v1.global_variables_initializer())

        #--------------------
        #       Train
        #--------------------
        loss = []
        for epoch in range(n_epochs):
            epoch_loss = []
            batch_gen = self.batch_generator(model.batch_size, model_name = model.nn_type)
            _b = 0 # batch counter

            for batch in batch_gen:  
                imgs, labels = batch
                feed_val = [model.summary, model.loss, model.reconstr_loss , model.latent_loss, model.train]
                feed_dict = {model.x: imgs, model.y: labels} 
                
                summary, ELBO_loss, reconstr_loss , latent_loss, _ = model.sess.run(feed_val, feed_dict=feed_dict)
                
                batch_loss = [ELBO_loss, np.mean(reconstr_loss), np.mean(latent_loss)]
                epoch_loss.append(batch_loss)
                
                model.writer.add_summary(summary, int(_b + epoch*model.batch_size))
                
                _b += 1                
                if _b == n_batches:
                    break

            epoch_loss = np.mean(epoch_loss, 0)
            loss.append(epoch_loss)

            log.info("epoch %i: loss %0.8f, reconstr loss %0.8f, latent loss %0.8f"
                        % (epoch, epoch_loss[0], epoch_loss[1], epoch_loss[2]))    
            

            # Save progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, epoch))
                np.savez("%s/progress-%i-acc.npz" % (save_path, epoch), loss=np.array(loss),
                        learning_rate=model.learning_rate, epoch=n_epochs, batch_size=model.batch_size,
                        clip=model.max_grad_norm)

        # Save the final model                
        model.saver.save(model.sess, "%s/model.ckpt" % save_path)
        np.savez("%s/acc.npz" % save_path, loss=np.array(loss), learning_rate=model.learning_rate, epoch=n_epochs,
                batch_size=model.batch_size, clip=model.max_grad_norm)




