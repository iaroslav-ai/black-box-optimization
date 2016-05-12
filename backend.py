'''
Optimize function which does not change (too much) 
with its every new evaluation

Initialization: with bounds. These can be real/integer interval or category

@author: iaroslav
'''

import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import heapq

def test_method(method, bounds, objective, N):
    """
    This tests how well certain optimization black box optimization method
    optimizes certain objective func. w.r.t. given bounds and limit of N 
    evaluations of the objective
    """
    
    data = extend_data(None, None, None)
    
    for i in range(N):
        choice, tosave = method(data, bounds)
        #print choice
        val = objective(choice)
        data = extend_data(data, tosave, val)
        
        res = select_best(data)
        print method.__name__, "iter", i, 'perf:', res[1]
       
        
    res = select_best(data)
    return res[1]    

def space_size(bounds):
    """ calculates total number of elements in space defined by bounds. 
    Optimization for small search spaces.
    """
    
    spacesize = 1
    
    for b in bounds.values():
        if b['type'] == 'real':
            spacesize = spacesize * 1000
        elif b['type'] == 'integer':
            spacesize = spacesize * (b['bound'][1] - b['bound'][0])
        else:
            spacesize = spacesize * len(b['bound'])
    
        if spacesize > (2 ** 100):
            # unlikely to be enumerated completely
            return spacesize;
    
    return spacesize

"""
The code below is needed to manage observed values of objective
"""

def extend_data(data, point, value):
    if data is None:
        return [[None, -1e10]]
    data.append([point, value])
    return data

def select_best(data):
    i, _ = max(enumerate(data), key = lambda p: p[1][1])
    return data[i]

def random_sample(bounds):
    choice = {}
        
    for n in bounds:
        b = bounds[n]
        if b['type'] == 'real':
            choice[n] = random.uniform(b['bound'][0],b['bound'][1])
        elif b['type'] == 'integer':
            choice[n] = random.randint(b['bound'][0],b['bound'][1])
        else:
            choice[n] = random.choice(b['bound'])
    
    return choice

def random_choice(data, bounds):
    """Randomly sample element from space defined by bounds, until it is not one
    of the previous samples stored in data array.
    """
    
    if len(data) >= space_size(bounds):
        return None;
       
    choice = None
    
    while str(choice) in [str(d[0]) for d in data]:
        choice = random_sample(bounds)
    
    return choice, choice

def genetic_choice(data, bounds):
    """Selects small subset of the points with best objective value, and either
    mix them or mutate them, and returns a result.
    """
    
    if len(data) >= space_size(bounds):
        return None;
       
    choice = None
    best = heapq.nlargest(3, data, key = lambda p: p[1])
    
    while str(choice) in [str(d[0]) for d in data]:
        # generate choice           
        action = random.choice(["mix","mutate"])
        
        if action == "mix":
            # pick two random candidates from pool and mix them
            A = random.choice(best)[0]
            B = random.choice(best)[0]
        else:
            # pick one random candidate and mutate it
            A = random.choice(best)[0]
            B = random_sample(bounds)
        
        if A is None or B is None:
            choice = random_sample(bounds)
        else:
            choice = { k: random.choice([ A[k], B[k] ]) for k in A }
        
    return choice, choice

### ==================== BO code. Madness starts here ======================

def to_features(v, bounds):
    x = []
    
    for n in bounds:
        b = bounds[n]
        if b['type'] == 'real':
            x.append( v[n] )
        elif b['type'] == 'integer':
            x.append( v[n] )
        else:
            x.append( b['bound'].index(v[n]) )
            
    return x

def from_features(v, bounds):
    x = {}
    idx = 0
    for n in bounds:
        b = bounds[n]
        if b['type'] == 'real':
            x[n] = v[idx]
        elif b['type'] == 'integer':
            x[n] = int(v[idx])
        else:
            c = np.clip(v[idx], 0, len(b['bound'])-1)
            x[n] = b['bound'][ int(c) ]
        idx = idx + 1
            
    return x

def bound_rnd_feat(bounds):
    v = random_sample(bounds)
    return to_features(v, bounds)

def lower_upper_bound(bounds):
    # generates bounds on feature values
    lower, upper = {}, {}
        
    for n in bounds:
        b = bounds[n]
        lower[n]= b['bound'][0]
        upper[n]= b['bound'][-1]
    
    return to_features(lower, bounds), to_features(upper, bounds)

def fit_gauss_proc(X,Y,C):
    
    X = np.array(X)
    Y = np.array(Y)
    
    # fit gaussian process to the data
    sq_dist = squareform(pdist(X, 'euclidean')) ** 2
    K = np.exp(-C*sq_dist )
    try:
        Kinv = np.linalg.inv(K)
    except BaseException as ex:
        print ex
    Kf = np.dot(Kinv, Y);
    
    return Kinv, Kf

def predict_gauss_proc(X, Xp, Kinv, Kf, C):
    
    m, d = [], []
    for x in Xp:
        sq_dist = np.sum( (X - x)**2 , axis = 1)
        k = np.exp(-C*sq_dist)
        mu = np.dot(k, Kf)
        sg = 1 - np.dot(k, np.dot(Kinv, k))
        m.append(mu)
        d.append(sg)
    return np.array(m), np.array(d)

def crossval_gauss_proc(X,Y,C):
    sp = len(X) / 2
    
    X = np.array(X)
    Y = np.array(Y)
    
    Xa, Xb = X[:sp], X[sp:]
    Ya, Yb = Y[:sp], Y[sp:]
    
    Kinv, Kf = fit_gauss_proc(Xa, Ya, C)
    Mb, _ = predict_gauss_proc(Xa, Xb, Kinv, Kf, C)
    Sb = np.mean( (Yb - Mb) ** 2 )
    
    Kinv, Kf = fit_gauss_proc(Xb, Yb, C)
    Ma, _ = predict_gauss_proc(Xb, Xa, Kinv, Kf, C)
    Sa = np.mean( (Ya - Ma) ** 2 )
    
    return np.mean([Sa, Sb])
    

def fit_optimize_gauss_proc(X,Y,bounds):
    # trains guassian process model on the data
    # makes prediction for next input x to try
    
    # fit GP
    Cs = []
    
    rng = 10**np.linspace(-3, 4, 100)
    
    for c in rng:
        Cs.append( [crossval_gauss_proc(X, Y, c), c] )
    
    i, _ = min( enumerate(Cs), key = lambda p: p[1][0] )
    
    #print Cs
    #print Cs[i][0]
    C = Cs[i][1]
    
    explore = (len(X) % 3) / 2.0
    #print explore, C
    
    Kinv, Kf = fit_gauss_proc(X, Y, C)
    
    # define exploration fnc
    def val(x): 
        sq_dist = np.sum( (X - x)**2 , axis = 1)
        k = np.exp(-C*sq_dist)
        mu = np.dot(k, Kf)
        sg = 1 - np.dot(k, np.dot(Kinv, k))
        return -(mu + sg*explore) # we do maximization
    
    # get bounds for params values
    l,u = lower_upper_bound(bounds)
    X = np.array(X)
    
    bestv = -1e10;
    bestx = None;
    
    # optimize starting from some random points
    for i in range(5):
    
        x0 = bound_rnd_feat(bounds)
        # maximization = minimization of negative objective
        bv = zip(l,u)
        sol = minimize(val, x0, method="L-BFGS-B", bounds = bv, tol = 1e-9)
        x = sol.x
        f = val(x)
        
        # this is needed to avoid num. problems with fitting GP to data later
        check = np.min( np.sum(np.abs(X - x), axis = 1) ) > 1e-2 
        
        if check and bestv < f:
            bestx = x
            bestv = f
    
    if bestx is None:
        bestx = np.array( bound_rnd_feat(bounds) )
    
    return bestx

def bayesian_choice(data, bounds):
    # in few 1,2 iteration, simply choose boundary and center points.
    # in later iterations, choose by optimizing 

    X = [v[0] for v in data if not v[0] is None]
    Y = [v[1] for v in data if not v[0] is None]
    
    if len(X) == 0:
        l,u = lower_upper_bound(bounds)
        c = from_features(l, bounds)
        return c, l
    
    if len(X) == 1:
        l,u = lower_upper_bound(bounds)
        c = from_features([ lv*0.5 + uv*0.5 for lv, uv in zip(l,u) ], bounds)
        return c, u
    
    x = None
    
    while str(x) in [str(d[0]) for d in data]:
        x = fit_optimize_gauss_proc(X, Y, bounds)
        
    return from_features(x, bounds), x.tolist()

    