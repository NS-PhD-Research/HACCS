import os
import tarfile
import shutil
import numpy as np

import urllib.parse

from urllib.request import urlretrieve
from dataset import Dataset

######constants######
# Absolute path of "data" directory 
DATADIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

class CIFAR10(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.path = os.path.join(DATADIR, 'cifar10')
        self.url = 'https://www.cs.toronto.edu/~kriz/'
        self.tar = 'cifar-10-binary.tar.gz'
        self.files = [  'cifar-10-batches-bin/data_batch_1.bin',
                        'cifar-10-batches-bin/data_batch_2.bin',
                        'cifar-10-batches-bin/data_batch_3.bin',
                        'cifar-10-batches-bin/data_batch_4.bin',
                        'cifar-10-batches-bin/data_batch_5.bin',
                        'cifar-10-batches-bin/test_batch.bin']
                        
    class Factory:
        def get(self):
            return CIFAR10()
    
    def download_data(self):
        # Create cifar10 directory is non-existent
        os.makedirs(self.path, exist_ok=True)

        # Download cifar10 tar file is non-existent
        if self.tar not in os.listdir(self.path):
            urlretrieve(urllib.parse.urljoin(self.url, self.tar), os.path.join(self.path, self.tar))
            print('Downloaded ', self.tar, ' from ', self.url)
        
        # Retrieve train and test data from tar file
        with tarfile.open(os.path.join(self.path, self.tar)) as tarobj:
            # Each file has 10000 color (32*32*3) images and 10000 labels
            fsize = 10000 * (32 * 32 * 3) + 10000

            # 6 files with pixel values in range(0, 255): uint8
            imgbin = np.zeros(fsize * 6, dtype='uint8')

            # Get the name of 6 binary files from tar object
            binNames = [file for file in tarobj if file.name in self.files]

            # To order train and test data, sort the binNames based on name
            binNames.sort(key=lambda file: file.name)

            for iter, file in enumerate(binNames):
                # Handle to binary file
                binHandle = tarobj.extractfile(file)

                # Read binary data in bytes to imgbin
                imgbin[iter * fsize : (iter + 1) * fsize] = np.frombuffer(binHandle.read(), dtype='B')
        
        # remove the cifar10 directory after use
        # shutil.rmtree(self.path)

        # 1 image : 3073 bytes [1(label):32*32*3(rbg)]
        labels = imgbin[::3073]

        # Remove the labels to get the pixels of all images
        pixels = np.delete(imgbin, np.arange(0, imgbin.size, 3073))

        # Convert into separate flat images (1*3072)
        imgflat = pixels.reshape(-1, 3072).astype('float32')/255

        # Convert flat image to channel * width * height followed by transpose for use by NN tensors
        # Reference: https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
        imgReshape = imgflat.reshape(-1, 3, 32, 32)
        img = imgReshape.transpose(0, 2, 3, 1)

        # Train and test data
        self.train_x, self.train_y = img[:50000], labels[:50000]
        self.test_x, self.test_y = img[50000:], labels[50000:]

        # get unique label count
        self.unique_labels = list(np.unique(self.train_y))
        self.n_unique_labels = len(self.unique_labels)
        self.min_label = min(self.unique_labels)
        self.max_label = max(self.unique_labels)

        # list of list: for both train and test
        #           inner list: indices corresponding to a specific label
        #           outer list: unique labels
        self.indices_train = [[] for x in range(self.n_unique_labels)]
        self.indices_test = [[] for x in range(self.n_unique_labels)]
        for i in range(self.n_unique_labels):
            self.indices_train[i] = np.isin(self.train_y, [i])
        for i in range(self.n_unique_labels):
            self.indices_test[i] = np.isin(self.test_y, [i])

        return True

    def get_training_data(self, id):
        return super().get_training_data(id)
    
    def get_testing_data(self, id):
        return super().get_testing_data(id)


if __name__ == '__main__':

    cls = CIFAR10()
    cls.download_data()
