'''
Created on Nov 14, 2015

@author: Iaroslav
'''

from mnist_py import MNIST

# example usage of MNIST loader:
data = MNIST(verbose=True);

# this are the contents of the loaded data:
print data.train_images.shape
print data.train_labels.shape

print data.test_images.shape
print data.test_labels.shape
