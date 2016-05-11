'''
This performs tests of optimization procedures
@author: iaroslav
'''

import example_problems as ex
import backend as bc
import numpy as np

def test_method(method, bounds, obj, N):
    
    # initialize
    data = bc.extend_data(None, None, None)
    
    for i in range(N):
        choice, tosave = method(data, bounds)
        #print choice
        val = obj(choice)
        data = bc.extend_data(data, tosave, val)
        
        res = bc.select_best(data)
        print method.__name__, "iter", i, 'perf:', res[1]
       
        
    res = bc.select_best(data)
    return res[1]    

totest = [bc.bayesian_choice, bc.genetic_choice, bc.random_choice]

tests = [ex.MnistSubsetLinearClassification]

for testc in tests:
    N = 5
    res = []
    
    test = testc()
    print test.name
    
    for rep in range(10):
        test = testc()
        sample = [test_method(m, test.bounds, test.obj, N) for m in totest]
        
        print 'iter', rep, sample
        
        res.append(sample)
    
    # print median score - more better
    print np.median(res, axis = 0)
    # print median deviation - less is better
    print np.median( np.abs( np.array(res) - np.median(res, axis = 0) ), axis = 0 )

