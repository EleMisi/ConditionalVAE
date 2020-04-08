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
        self.labels, self.attr = self.create_labels()
        self.train_labels, self.test_labels = self.split() 
        self.n_train = len(self.train_labels)
        self.n_attr = len(self.attr)
        

    def get_image(self, image_path, img_size = 128, img_resize = 64, x = 25, y = 45):
        """
        Return an image as a flattened normalized numpy array (dim 64*64*3)
        """
        mode='RGB' 
        image = Image.open(image_path)
        # Crop 
        image = image.crop([x, y, x + img_size, y+img_size])
        # Resize
        image = image.resize([img_resize, img_resize], Image.BILINEAR)
        # Normalization
        img = np.array(image.convert(mode)).astype('float32')
        img = img.ravel()
        img /= 255.

        return np.array(img)


    def create_image_dataset(self, labels):
        """
        Return an List with the images corresponding to the given labels.
        The images are normalized and returned as a raveled array.
        """
        imgs = []
        images_dataset = glob('/input/CelebA/img_align_celeba/img_align_celeba/*.jpg')
        imgs_id = [item[0] for item in labels]

        for i in images_dataset:
            if os.path.split(i)[1] in imgs_id:
              imgs.append(self.get_image(image_path = i))

        return imgs

    def create_txt_dataset(self):
        """
        Return
            labels OrderedDict with:
                Key = image number
                Value = List of attributes with value (0,1)
            attr_list List with attributes names
        """
        file_path = "/input/CelebA/list_attr_celeba.csv"
        df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)

        attr_list = [x for x in df.columns] 
        od = OrderedDict(df.to_dict('index'))
        labels = OrderedDict()
        for k,v in od.items():
          label = [x for x in v.values()]
          labels[k] = label

        print(len(labels), len(attr_list))

        return labels, attr_list

    def create_labels(self):
        """
        Return:
            labels OrderedDict with:
                Key = image number
                Value = List of attributes with value (0,1)
            attr List with attributes names

        """
        print("Loading labels and attributes names")
        labels, attr = self.create_txt_dataset()
        return labels, attr


    def split(self):
        """
        Prepare the train set and the test set, according to 
        train_dim.
        Returned items are lists of tuples [(im_id1, label_1), ..., (im_idN, label_N)]
        """
        print("\n-------Splitting dataset------\n")

        # Shuffle 
        shuffled_labels = self.shuffle()
        # Split 
        n_train = int(len(self.labels) * self.train_dim)
        list_items = list(shuffled_labels.items())
        train_labels = list_items[:n_train]
        test_labels = list_items[n_train:]

        print("Train set dimension (%i), test set dimension (%i) \n" % (len(train_labels), len(test_labels)))
        
        return train_labels, test_labels
    


    def shuffle(self):
        """
        Return shuffled OrderedDict of labels
        """
        items = list(self.labels.items())
        random.shuffle(items)
        shuffled_labels = OrderedDict(items)
    
        return shuffled_labels

    def next_batch(self, batch_dim, data, labels):
        """
        Create the next batch with a given dimension.
        -data and labels are lists 
        """
        #Shuffle 
        shuffled_data, shuffled_labels = self.shuffle()
        #Create batch
        batch_data = shuffled_data[:batch_dim]
        batch_labels = shuffled_labels[:batch_dim]

        return np.asarray(batch_data), np.asarray(batch_labels)

    def batch_generator(self, batch_dim):
        """
        Batch generator using the label OrderedDict
        """

        while True:
            batch_imgs = []
            labels = []
            for label in (self.train_labels):
                labels.append(label)
                if len(labels) == batch_dim:
                    batch_imgs = self.create_image_dataset(labels)
                    batch_labels = [x[1] for x in labels]
                    yield (np.asarray(batch_imgs), np.asarray(batch_labels))
                    batch_imgs = []
                    labels = []
                    batch_labels = []
            if batch_imgs:
                yield (np.asarray(batch_imgs), np.asarray(batch_labels))

    #------------------------------------
    #------------CelebA Train------------
    #------------------------------------

    def celebA_train(self, model, epoch, save_path="./"):
        """
        Training function for CelebA dataset.
        """
        n_train = self.n_train
        n_batches = int(n_train / model.batch_size)

        # Log
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        log = create_log(save_path+"Log")
        log.info("Training set dimension(%i) batch num(%i), batch size(%i)" % (n_train, n_batches, model.batch_size))
        results = []

        # Session initialization
        model.sess.run(tf.compat.v1.global_variables_initializer())

        #-----------------Train-----------------
        for _e in range(epoch):
            _results = []
            batch_gen = self.batch_generator(model.batch_size)
            _b = 0 #counter
            for batch in batch_gen:  
                _x, _y = batch
                feed_val = [model.summary, model.loss, model.reconstr_loss , model.latent_loss, model.train]
                feed_dict = {model.x: _x, model.y: _y} 
                summary, loss, reconstr_loss , latent_loss, _ = model.sess.run(feed_val, feed_dict=feed_dict)
                __result = [loss, reconstr_loss , latent_loss]
            
                _results.append(__result)
                model.writer.add_summary(summary, int(_b + _e * model.batch_size))
                _b += 1
                print(_b)
                if _b == n_batches:
                    break

            #---------------Validation--------------
            _results = np.mean(_results, 0)
            log.info("epoch %i: loss %0.3f, reconstr. loss %0.3f, latent loss %0.3f"
                        % (_e, _results[0], _results[1], _results[2]))    
            results.append(_results)

            #----------Save progress every 5 epochs----------
            if (_e + 1) % 5 == 0:
                model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
                np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(results),
                        learning_rate=model.learning_rate, epoch=epoch, batch_size=model.batch_size,
                        clip=model.max_grad_norm)

        #------------Save the final model--------------                 
        model.saver.save(model.sess, "%s/model.ckpt" % save_path)
        np.savez("%s/acc.npz" % save_path, loss=np.array(results), learning_rate=model.learning_rate, epoch=epoch,
                batch_size=model.batch_size, clip=model.max_grad_norm)




