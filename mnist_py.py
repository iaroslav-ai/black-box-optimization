'''
Created on Nov 14, 2015

@author: Iaroslav
'''

import scipy.io as sio
import numpy as np
from collections import namedtuple

def MNIST(verbose = False):
   
    if(verbose):print 'Started loading ... '
    
    MNIST_Dataset = namedtuple('MNIST_Dataset', ['train_images', 'train_labels', 'test_images', 'test_labels'])
    data = sio.loadmat('MNIST.mat');
    dataset = MNIST_Dataset( train_images = np.transpose(data['train_images']),
                             train_labels = data['train_labels'],
                             test_images = np.transpose(data['test_images']),
                             test_labels = data['test_labels']);

    if(verbose):print 'Loaded MNIST dataset'
    
    return dataset