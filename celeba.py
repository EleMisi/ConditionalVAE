import numpy as np


class CelebA():

    def __init__(self, train_dim):
        self.train_dim = train_dim
        self.data, self.labels, self.attr = self.load_dataset()
        self.dataset_dim = len(self.data)
        self.n_attr = len(self.attr)
        self.train_set, self.test_set, self.train_labels, self.test_labels = self.split()

    def load_dataset(self):
        """Return:
            -data: OrderedDict {img_id : img}
            -labels: OrderedDict {img_id : attributes}
            -attr: dict [attribute names : idx] 
            """
        pass
   
    def split(self):
        """
        Prepare the train set and the test set, according to 
        train_dimension.
        The sets are returned as lists
        """
        data = list(self.data.values())
        labels = list(self.labels.values())
        #Shuffle 
        shuffled_data, shuffled_labels = self.shuffle(data, labels)
        #Split 
        n_train = self.dataset_dim * self.train_dim
        train_set = shuffled_data[:n_train]
        train_labels = shuffled_labels[:n_train]
        test_set = shuffled_data[n_train:]
        test_labels = shuffled_labels[n_train:]

        return train_set, test_set, train_labels, test_labels
    


    def shuffle(self, data, labels):
        """
        Return shuffled data and related labels.
        -list data
        -list labels
        """
        idx = np.arange(0 , len(data))
        np.random.shuffle(idx)
        shuffled_data = [data[i] for i in idx]
        shuffled_labels = [labels[i] for i in idx]

        return shuffled_data, shuffled_labels

    def next_batch(self, batch_dim, data, labels):
        """
        Create the next batch with a given dimension.
        -data and labels are lists 
        """
        #Shuffle 
        shuffled_data, shuffled_labels = self.shuffle(data, labels)
        #Create batch
        batch_data = shuffled_data[:batch_dim]
        batch_labels = shuffled_labels[:batch_dim]

        return np.asarray(batch_data), np.asarray(batch_labels)