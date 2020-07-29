from collections import OrderedDict
import cv2
from tensorflow.keras.utils import Sequence
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

from utils import save_VarToSave



class CelebADataset(Sequence):
    """
    Keras Sequence for CelebA Dataset.
    """

    def __init__(self, train_size, batch_size, mode = 'train', save_test_set = False):

        self.train_labels, self.test_labels, self.attributes = self.load_labels(train_size)
        self.batch_size = batch_size
        self.mode = mode
        self.train_size = len(self.train_labels)
        if save_test_set:
            self.save_test_set()


    def load_labels(self, train_dim):

        print("Loading labels and attributes...")

        file_path = "/input/CelebA/list_attr_celeba.csv"
        df = pd.read_csv(file_path, header = 0, index_col = 0).replace(-1,0)
        attributes = [x for x in df.columns] 
        od = OrderedDict(df.to_dict('index'))
        labels = OrderedDict()
        for k,v in od.items():
          label = [np.float32(x) for x in v.values()]
          labels[k] = label
        print("Labels: {} \nAttributes: {} \n".format(len(labels), len(attributes)))

        #Splitting
        print("Splitting dataset...\n")
        n_train = int(len(labels) * train_dim)
        list_labels = list(labels.items())
        train_labels = list_labels[:n_train]
        test_labels = list_labels[n_train:]

        print("Train set dimension: {} \nTest set dimension: {} \n".format(len(train_labels), len(test_labels)))

        return train_labels, test_labels, attributes

    def next_batch(self, idx):
        """
        Returns a batch of images with their labels.
        """    

        batch_labels = [x[1] for x in self.train_labels[idx * self.batch_size : (idx + 1) * self.batch_size]]
        images_id = [x[0] for x in self.train_labels[idx * self.batch_size : (idx + 1) * self.batch_size]]
        batch_imgs = self.get_images(images_id) 
        return np.asarray(batch_imgs, dtype='float32'), np.asarray(batch_labels, dtype='float32')

    def preprocess_image(self,image_path, img_size = 128, img_resize = 64, x = 25, y = 45):
        """
        Crops, resizes and normalizes the target image.
        """

        img = cv2.imread(image_path)
        img = img[y:y+img_size, x:x+img_size]
        img = cv2.resize(img, (img_resize, img_resize))
        img = np.array(img, dtype='float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img /= 255.0

        return img

    def get_images(self,imgs_id):
        """
        Returns the list of images corresponding to the given labels.
        """
        imgs = []

        for i in imgs_id:
            image_path ='/input/CelebA/img_align_celeba/img_align_celeba/' + i
            imgs.append(self.preprocess_image(image_path))

        return imgs

    def save_test_set(self):
        """
        Saves a dictionary with the test set information.
        """
        try:
            test_data = {
                'train_size' : self.train_size,
                'test_labels' : self.test_labels,
                'attributes' : self.attributes,
                'batch_size' : self.batch_size
            }

            file_path = "./test_data"
            save_VarToSave(file_path, test_data)
        except:
            raise
        print("Test labels successfully saved.")

    def shuffle(self):
        """
        Shuffles the train labels.
        """
        self.train_labels = random.sample(self.train_labels, k=self.train_size)
        print("Labels shuffled.")

    def __len__(self):
        return int(math.ceil(self.train_size / float(self.batch_size)))

    def __getitem__(self, index):
        return self.next_batch(index)
