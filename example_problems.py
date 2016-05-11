'''
This contains a set of test problems for global optimization algorithms
@author: iaroslav
'''
import math
import numpy as np
import random
from sklearn.svm import SVR, SVC, LinearSVC
from mnist_reader import MNIST

class MnistSubsetLinearClassification():

    def __init__(self):
        
        data = MNIST()
        
        # choose a subset of the labels ...
        labels = random.sample([0,1,2,3,4,5,6,7,8,9], 5)
        
        # choose points only with selected labels
        I = np.zeros_like(data.train_labels)
        
        for l in labels:
            I = I + (data.train_labels == l)*1.0
        
        I = I > 0.5
        I = I[:,0]
        
        # perform selection
        X = data.train_images[I,:]
        Y = data.train_labels[I,0]
        
        sp = (len(X) // 4)*3 # 75% training, 25% validation
        # train validation split
        self.X, self.Xv =  X[:sp], X[sp:]
        self.Y, self.Yv =  Y[:sp], Y[sp:]
        
        self.name = "MNIST classification"
        self.bounds = {'C':{
                            'type':'real',
                            'bound':[-2.0, 3.0] # power of 10
                            },
                       'loss':{
                            'type':'category',
                            'bound':['hinge', 'squared_hinge']
                            },
                       'penalty':{
                            'type':'category',
                            'bound':['l1', 'l2']
                            },
                       'dual':{
                            'type':'category',
                            'bound':['True', 'False']
                            }
                       }
    
    def obj(self, cfg):
        
        try:
            
            # create the regressor with given params
            clsf = LinearSVC(C = 10.0 ** cfg['C'], 
                       loss= cfg['loss'], 
                       penalty=cfg['penalty'], 
                       dual=cfg['dual'] == 'True')
            
            # fit the regressor
            clsf.fit(self.X, self.Y)
            
            # get the validation score
            score = clsf.score(self.Xv, self.Yv)
            
            return score
        
        except BaseException:
            return 0.0

class MnistSubsetClassification():

    def __init__(self):
        
        data = MNIST()
        
        # choose a subset of the labels ...
        labels = random.sample([0,1,2,3,4,5,6,7,8,9], 3)
        
        # choose points only with selected labels
        I = np.zeros_like(data.train_labels)
        
        for l in labels:
            I = I + (data.train_labels == l)*1.0
        
        I = I > 0.5
        I = I[:,0]
        
        # perform selection
        X = data.train_images[I,:]
        Y = data.train_labels[I,0]
        
        sp = (len(X) // 4)*3 # 75% training, 25% validation
        # train validation split
        self.X, self.Xv =  X[:sp], X[sp:]
        self.Y, self.Yv =  Y[:sp], Y[sp:]
        
        self.name = "MNIST classification"
        self.bounds = {'C':{
                            'type':'real',
                            'bound':[-2.0, 3.0] # power of 10
                            },
                       'gamma':{
                            'type':'real',
                            'bound':[-2.0, 1.0] # power of 10
                            },
                       'kernel':{
                            'type':'category',
                            'bound':['rbf', 'linear']
                            }
                       }
    
    def obj(self, cfg):
        # create the regressor with given params
        clsf = SVC(C = 10.0 ** cfg['C'], 
                   gamma= 10.0 ** cfg['gamma'], 
                   kernel=cfg['kernel'], 
                   verbose = True)
        
        # fit the regressor
        clsf.fit(self.X, self.Y)
        
        # get the validation score
        score = clsf.score(self.Xv, self.Yv)
        
        return score

class ArtificialRegression():
    def __init__(self):
        
        # size of data
        self.N = 200
        self.M = 5 # features
        
        # generate random data ...
        self.X = np.random.randn(self.N, self.M)
        self.Xv = np.random.randn(self.N, self.M) # validation part
        
        # parameters of unknown relation between inputs and outputs ..
        self.W = np.random.randn(self.M)
        
        def relation(X):
            return np.sin( np.dot(X, self.W) )
        
        self.Y = relation(self.X)
        self.Yv = relation(self.Xv)        
        
        self.name = "Nonlinear fit"
        self.bounds = {'C':{
                            'type':'real',
                            'bound':[-2.0, 3.0] # power of 10
                            },
                       'epsilon':{
                            'type':'real',
                            'bound':[-2.0, 0.0] # power of 10
                            },
                       'gamma':{
                            'type':'real',
                            'bound':[-2.0, 1.0] # power of 10
                            },
                       'kernel':{
                            'type':'category',
                            'bound':['rbf', 'linear']
                            }
                       }
    
    def obj(self, cfg):
        # create the regressor with given params
        clsf = SVR(C = 10.0 ** cfg['C'], 
                   epsilon= 10.0 ** cfg['epsilon'], 
                   gamma= 10.0 ** cfg['gamma'], 
                   kernel=cfg['kernel'])
        
        # fit the regressor
        clsf.fit(self.X, self.Y)
        
        # get the validation score
        score = clsf.score(self.Xv, self.Yv)
        
        return score
