from collections import OrderedDict
import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import sys
import tensorflow as tf

# from utils import create_log
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
        # Normalization: Convert images to floating point with the range [-0.5, 0.5]
        img = np.array(image.convert(mode)).astype('float32')
        img /= 255.0
        # Data Augmentation
        if random.random() < 0.5:
            img = np.flip(img, 1)

        if model_name == "Dense" :
            img = img.ravel()
        
        return np.array(img)


    def create_image_batch(self, imgs_id, model_name):
        """
        Returns the list of images corresponding to the given labels.
        """
        imgs = []

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
        print("Loading labels and attributes...")

        file_path = "/input/CelebA/list_attr_celeba.csv"
        df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)
        # Remove unuseful attributes
        df.drop(["Wearing_Necklace", "Wearing_Necktie"], axis=1, inplace = True)
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
        #shuffled_labels = self.shuffle()
        n_train = int(len(self.labels) * self.train_dim)
        labels = OrderedDict(list(self.labels.items()))
        list_labels = list(labels.items())
        # Split (according to train_dim)
        train_labels = list_labels[:n_train]
        test_labels = list_labels[n_train:]

        print("Train set dimension: {} \nTest set dimension: {} \n".format(len(train_labels), len(test_labels)))
        
        return train_labels, test_labels
    


    def batch_generator(self, batch_dim, model_name):
        """
        Batch generator.
        """
        while True:
            batch_imgs = []
            batch_labels = []
            images_id = []
            labels = self.train_labels
            # Shuffling
            random.shuffle(labels)

            for i in range(batch_dim):
               idx = random.randint(0,len(self.train_labels)-1)
               batch_labels.append(labels[idx][1])
               images_id.append(labels[idx][0])

            batch_imgs = self.create_image_batch(images_id, model_name) 

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




