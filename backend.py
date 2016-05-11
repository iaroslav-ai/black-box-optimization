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

def spsz(bounds):
    
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

def extend_data(data, point, value):
    if data is None:
        return [[None, -1e10]]
    data.append([point, value])
    return data

def select_best(data):
    i, _ = max(enumerate(data), key = lambda p: p[1][1])
    return data[i]

def sel_rnd(bounds):
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
    """Returns the next choice given pervious data and bounds
    """
    
    if len(data) >= spsz(bounds):
        return None;
       
    choice = None
    
    while str(choice) in [str(d[0]) for d in data]:
        choice = sel_rnd(bounds)
    
    return choice, choice

def genetic_choice(data, bounds):
    """Returns the next choice given pervious data and bounds
    """
    
    if len(data) >= spsz(bounds):
        return None;
       
    choice = None
    best = heapq.nlargest(5, data, key = lambda p: p[1])
    
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
            B = sel_rnd(bounds)
        
        if A is None or B is None:
            choice = sel_rnd(bounds)
        else:
            choice = { k: random.choice([ A[k], B[k] ]) for k in A }
        
    return choice, choice

### ==================== BO ======================

def b_to_f(v, bounds):
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

def f_to_b(v, bounds):
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
    v = sel_rnd(bounds)
    return b_to_f(v, bounds)

def lower_upper_bound(bounds):
    # generates bounds on feature values
    lower, upper = {}, {}
        
    for n in bounds:
        b = bounds[n]
        lower[n]= b['bound'][0]
        upper[n]= b['bound'][-1]
    
    return b_to_f(lower, bounds), b_to_f(upper, bounds)

def fit_gp(X,Y,C):
    
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

def prd_gp(X, Xp, Kinv, Kf, C):
    m, d = [], []
    for x in Xp:
        sq_dist = np.sum( (X - x)**2 , axis = 1)
        k = np.exp(-C*sq_dist)
        mu = np.dot(k, Kf)
        sg = 1 - np.dot(k, np.dot(Kinv, k))
        m.append(mu)
        d.append(sg)
    return np.array(m), np.array(d)

def gp_cvl(X,Y,C):
    sp = len(X) / 2
    
    X = np.array(X)
    Y = np.array(Y)
    
    Xa, Xb = X[:sp], X[sp:]
    Ya, Yb = Y[:sp], Y[sp:]
    
    Kinv, Kf = fit_gp(Xa, Ya, C)
    Mb, _ = prd_gp(Xa, Xb, Kinv, Kf, C)
    Sb = np.mean( (Yb - Mb) ** 2 )
    
    Kinv, Kf = fit_gp(Xb, Yb, C)
    Ma, _ = prd_gp(Xb, Xa, Kinv, Kf, C)
    Sa = np.mean( (Ya - Ma) ** 2 )
    
    return np.mean([Sa, Sb])
    

def sel_bo(X,Y,bounds):
    # trains guassian process model on the data
    # makes prediction for next input x to try
    
    Cs = []
    
    rng = 10**np.linspace(-3, 4, 100)
    
    for c in rng:
        Cs.append( [gp_cvl(X, Y, c), c] )
    
    i, _ = min( enumerate(Cs), key = lambda p: p[1][0] )
    
    #print Cs
    #print Cs[i][0]
    C = Cs[i][1]
    
    explore = (len(X) % 3) / 2.0
    #print explore, C
    
    Kinv, Kf = fit_gp(X, Y, C)
    
    # define exploration fnc
    def val(x): 
        sq_dist = np.sum( (X - x)**2 , axis = 1)
        k = np.exp(-C*sq_dist)
        mu = np.dot(k, Kf)
        sg = 1 - np.dot(k, np.dot(Kinv, k))
        return -(mu + sg*explore) # we do maximization
    
    
    l,u = lower_upper_bound(bounds)
    X = np.array(X)
    
    bestv = -1e10;
    bestx = None;
    
    for i in range(5):
    
        x0 = bound_rnd_feat(bounds)
        # maximization = minimization of negative objective
        bv = zip(l,u)
        sol = minimize(val, x0, method="L-BFGS-B", bounds = bv, tol = 1e-9)
        x = sol.x
        f = val(x)
        
        sng_ch1 = np.min( np.sum(np.abs(X - x), axis = 1) ) > 1e-2 
        
        if sng_ch1 and bestv < f:
            bestx = x
            bestv = f
    
    if bestx is None:
        bestx = np.array( bound_rnd_feat(bounds) )
    
    return bestx

def bayesian_choice(data, bounds):

    X = [v[0] for v in data if not v[0] is None]
    Y = [v[1] for v in data if not v[0] is None]
    
    if len(X) == 0:
        l,u = lower_upper_bound(bounds)
        c = f_to_b(l, bounds)
        return c, l
    
    if len(X) == 1:
        l,u = lower_upper_bound(bounds)
        c = f_to_b([ lv*0.5 + uv*0.5 for lv, uv in zip(l,u) ], bounds)
        return c, u
    
    x = None
    
    while str(x) in [str(d[0]) for d in data]:
        x = sel_bo(X, Y, bounds)
        
    return f_to_b(x, bounds), x.tolist()

    