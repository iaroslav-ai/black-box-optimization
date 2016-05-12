'''
This performs tests of optimization procedures
@author: iaroslav
'''

import example_problems as ex
import backend as bc
import numpy as np

# methods of black box optimization to try
totest = [bc.bayesian_choice, bc.genetic_choice, bc.random_choice]

# classes of test problems to solve
tests = [ ex.ArtificialRegression, ex.MnistSubsetLinearClassification]

for testc in tests:
    
    N = 20 # max number of evals
    R = 10 # number of samples from certain test distribution to take
    
    res = []
        
    for rep in range(R):
        test = testc()
        sample = [bc.test_method(m, test.bounds, test.obj, N) for m in totest]
        
        print 'iter', rep, sample
        
        res.append(sample)
    
    print "Results for test ", test.name
    # print median score - more better
    print np.median(res, axis = 0)
    # print median deviation - less is better
    print np.median( np.abs( np.array(res) - np.median(res, axis = 0) ), axis = 0 )

