import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import gzip

FASHION_MNIST_CLASSES = ('T-shirt', 'Trouser', 'Pullover',
                         'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot')


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


class fashion_mnist_dataset(data.Dataset):
    def __init__(self, split, transform):
        self.split = split
        self.num_training = 50000
        self.num_validation = 10000
        self.num_test = 10000
        self.normalize = True
        self.transform = transform

        # Load the raw FASHION data
        if self.split == 'train':
            X, y = load_mnist('fashion-mnist', kind='train')
            mask = list(range(self.num_training))
            self.X = X[mask].astype(float)
            self.y = y[mask]
        elif self.split == 'val':
            X, y = load_mnist('fashion-mnist', kind='train')
            mask = list(range(self.num_training,
                        self.num_training + self.num_validation))
            self.X = X[mask].astype(float)
            self.y = y[mask]
        elif self.split == 'test':
            X, y = load_mnist('fashion-mnist', kind='t10k')
            mask = list(range(self.num_test))
            self.X = X[mask].astype(float)
            self.y = y[mask]

        '''
        if self.normalize:
            X_train, _ = load_mnist('fashion-mnist', kind='train')
            mask = list(range(self.num_training))
            X_train = X_train[mask].astype(float)

            mean_image = np.mean(X_train, axis=0)
            self.X -= mean_image
        '''

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        x = x.reshape((28, 28))
        x = torch.tensor(x).float().unsqueeze(0) / 255.

        y = self.y[index]
        y = torch.tensor(y).long()

        return x, y


if __name__ == "__main__":
    ds_val = fashion_mnist_dataset('val', None)

    x, y = ds_val[3]
    x = x.numpy()[0]

    plt.imshow(x)
    plt.show()
