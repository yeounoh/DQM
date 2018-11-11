import numpy as np
#from scipy.optimize import linprog
#from scipy.stats import poisson
import random, copy, math, pickle
import pylab as P


"""
    Baselines
"""
def nominal(data):
    return np.sum(np.sum(data==1, axis=1) > 0)

def vNominal(data):
    return np.sum(np.sum(data==1, axis=1) > np.sum(data != -1, axis=1)/2)

def sNominal(data,pos_switch=True,neg_switch=True):
    data_subset = data # no copying
    majority = np.zeros((len(data_subset),len(data_subset[0])))
    switches = np.zeros((len(data_subset),len(data_subset[0])))
    for i in range(len(data_subset)):
        prev = 0
        for w in range(0,len(data_subset[0])):
            # the first worker is compared with an algorithmic worker
            n_w = np.sum(data[i][0:w+1] != -1)
            n_pos = np.sum(data[i][0:w+1] == 1)
            n_neg = np.sum(data[i][0:w+1] == 0)

            maj = 0
            if n_pos == n_neg and n_pos != 0:
                # tie results in switch
                maj = (prev + 1)%2
            elif n_pos > n_w/2:
                maj = 1
            if prev != maj:
                if (maj == 1 and pos_switch) or (maj == 0 and neg_switch):
                    switches[i][w] = 1
            prev = maj
            majority[i][w] = maj

    return np.sum(np.logical_and(np.sum(switches,axis=1), np.sum(data,axis=1) != -1*len(data[0])))

"""
    Species estimation-based estimator
"""
def chao92(data):
    data_subset = copy.deepcopy(data)
    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)

    data_subset[data_subset == -1] = 0

    hist = np.sum(data_subset,axis=1)
    n = float(np.sum(hist))
    n_bar = float(np.mean([i for i in hist if i > 0]))
    v_bar = float(np.var(hist[hist > 0]))
    d = float(np.sum(hist > 0))
    f1 = float(np.sum(hist == 1))

    if n == 0:
        return d

    c_hat = max(1. - f1/n, 0.)
    if c_hat == 0.:
        return d

    gamma = coeff_of_variance(hist)
    return d/c_hat + n*(1-c_hat)/c_hat*gamma
  
def unseen(data):
    grid_factor = 1.05 #x_i (the grid of prob) will be geometric with this ratio
    alpha = .5 #avoid overfitting, smaller value increase the risk
    max_itr = 1000

    #data pre-processing
    data_subset = copy.deepcopy(data)
    pos_idx = np.sum(data == 1, axis=1) > np.sum(data != -1, axis=1)/2

    #discard opposing votes
    for i in range(len(data_subset)):
        if pos_idx[i]:
            data_subset[i,data_subset[i,:] == 0] = -1
        else:
            data_subset[i,data_subset[i,:] == 1] = -1

    n_worker = []
    for i in range(len(data_subset)):
        n_worker.append(np.sum(data_subset[i] != -1))
    n_worker = np.array(n_worker)

    #clean and no-ops are ignored; errorneous pairs are of different classes
    f = []
    n = 0
    for w in range(len(data_subset[0])):
        fx = np.sum(np.sum(data_subset == 1,axis=1) == w+1)
        f.append( fx )
        n += fx * (w+1)
    f = np.array(f)
    f1 = f[0] #zero-indexed

    #minimum allowable probability
    xLP_min = 1./(n*max(10,n))
    i_min = np.argmax(f > 0)
    if i_min > 0:
        xLP_min = (i_min+1.)/n

    #split the f-statistics into the dense portion and the sparse portion
    x= [0.]
    histx = [0.];
    fLP = np.zeros(len(f))
    for i in range(len(f)):
        if f[i] > 0:
            i_lower = int(max(0,i-math.ceil(math.sqrt(i))))
            i_upper = int(min(len(f)-1, i+math.ceil(math.sqrt(i))))
            if np.sum(f[i_lower:i_upper+1]) < math.sqrt(i):
                # sparse region used the empirical histogram
                x.append((i+1)/n)
                histx.append(f[i])
                fLP[i] = 0
            else:
                # will use LP for dense region
                fLP[i] = f[i]
    x = np.array(x)
    histx = np.array(histx)

    # no LP portion
    if np.sum(fLP > 0) == 0:
        x = x[1:]
        histx = histx[1:]
        return np.sum(histx)

    # first LP problem
    LP_mass = 1 - np.sum(x*histx)
    f_max = len(f) - np.argmax(fLP[::-1] > 0) - 1
    fLP = np.append(fLP[0:f_max+1],np.zeros(int(math.ceil(math.sqrt(f_max)))))
    szLPf = len(fLP)

    xLP_max = (f_max+1)/float(n)
    xLP = xLP_min*grid_factor**np.array( range( int(math.ceil(math.log(xLP_max/xLP_min)/math.log(grid_factor)))+1 ) )
    szLPx = len(xLP)

    objf = np.zeros(szLPx+2*szLPf)
    objf[szLPx::2] = 1./np.vectorize(math.sqrt)(fLP+1)
    objf[szLPx+1::2] = 1./np.vectorize(math.sqrt)(fLP+1)

    A = np.zeros((2*szLPf,szLPx+2*szLPf))
    b = np.zeros((2*szLPf,1))
    for i in range(szLPf):
        A[2*i,0:szLPx] = np.vectorize(lambda x:poisson.pmf(i+1,x))(n*xLP)
        A[2*i+1,0:szLPx] = -1 * A[2*i,0:szLPx]
        A[2*i,szLPx+2*i] = -1
        A[2*i+1,szLPx+2*i+1] = -1
        b[2*i] = fLP[i]
        b[2*i+1] = fLP[i]

    Aeq = np.zeros(szLPx + 2*szLPf)
    Aeq[0:szLPx] = xLP
    beq = LP_mass

    for i in range(szLPx):
        A[:,i] = A[:,i]/xLP[i]
        Aeq[i] = Aeq[i]/xLP[i]
    #result consists of x, slack, success, status, nit, message
    result = linprog(objf,A_ub=A,b_ub=b,A_eq=Aeq.reshape((1,len(Aeq))),b_eq=beq,options={'maxiter':max_itr})
    sol = result.x
    val = result.fun #objf_val = objf * sol
    #print 'first optimization result:',result.success,result.status,result.message

    # second LP problem
    objf2 = 0 * objf
    objf2[0:szLPx] = 1
    A2 = np.append(A,objf.reshape((1,len(objf))),axis=0)
    b2 = np.append(b, np.array(val)+alpha)
    for i in range(szLPx):
        objf2[i] = objf2[i]/xLP[i]
    result2 = linprog(objf2,A_ub=A2,b_ub=b2,A_eq=Aeq.reshape((1,len(Aeq))),b_eq=beq,options={'maxiter':max_itr})
    sol2 = result2.x
    #print 'second optimization result:',result2.success,result2.status,result2.message

    if not isinstance(sol2, np.ndarray):
        return np.sum(histx) 

    # combine the dense and sparse region solutions
    sol2[0:szLPx] = sol2[0:szLPx]/xLP
    x = np.append(x,xLP)
    histx = np.append(histx,sol2)
    idx = [i[0] for i in sorted(enumerate(x), key=lambda x:x[1])]
    x = x[idx]
    histx = histx[idx]
    x = x[histx > 0]
    histx = histx[histx > 0]

    return np.sum(histx)


"""
    Switch-based estimator 
"""
def switch(data):
    n_worker = len(data[0])
    est = vNominal(data)
    thresh = np.max([vNominal(data[:,:n_worker/2]), 
                     vNominal(data[:,:n_worker/4]), 
                     vNominal(data[:,:n_worker/4*3]) ])
    pos_adj = 0
    neg_adj = 0
    if est - thresh < 0:
        neg_adj = max(0,
                      remain_switch(
                        data,pos_switch=False,neg_switch=True) 
                        - sNominal(data,pos_switch=False,neg_switch=True)
                      )
    else:
        pos_adj = max(0,
                      remain_switch(
                        data,pos_switch=True,neg_switch=False) 
                        - sNominal(data,pos_switch=True,neg_switch=False)
                      )

    return max(0, est + pos_adj - neg_adj)

"""
    Triangular Walk
"""
def triangular_walk(data, n_max=3):
    n_items = len(data)

    linear_estimates = []
    for i in range(n_items):
        n_ = 0.
        k_ = 0.
        for w in data[i]:
            if w != -1:
                n_ += 1.
                k_ += float(w)

                if n_ < n_max and k_/n_ > 0.5:
                    continue
                else:
                    # check stopping conditions
                    if k_/n_ <= 0.5:
                        linear_estimates.append(0.)
                    else: # n=n_max
                        try:
                            if (2-n_max-2*k_)**2-4*(2*n_max-2)*k_ >= 0: 
                                p_ = ( 2.*k_+n_max-2+math.sqrt((2-n_max-2*k_)**2-4*(2*n_max-2)*k_)) / (4.*n_max-4)
                            else:
                                p_ = ((2.*k_+n_max-2)/(4*n_max-4))
                        except ValueError:
                            p_ = ((2.*k_+n_max-2)/(4*n_max-4))

                        linear_estimates.append(1. / (2*p_-1))
                    n_ = 0.
                    k_ = 0.
                    
    return np.mean(linear_estimates) * n_items
           
            
def expectation_maximization(data, alpha=0.8, beta=0.2):
    '''
        EM algorithm for worker quality estimation, argmax log(P(q|X, Beta~(alpha,beta))).
    '''
    
    # initialize with majority voting
    mu_clean = np.zeros(len(data))
    mu_dirty = np.zeros(len(data))
    for i in range(len(data)):
        if np.sum(data[i] == 1) > np.sum(data[i] != -1)/2.:
            mu_dirty[i] = 1.
        else:
            mu_clean[i] = 1.
    # initial worker quality set to alpha
    q_new = np.zeros(len(data[0])) + alpha
    q_ = np.zeros(len(data[0])) + 0.5
    while np.sum(q_new - q_) > 0.:
        q_ = q_new
        # E-step
        for i in range(len(data)):
            mu_err = 1.
            mu_not_err = 1.
            for j in range(len(data[i])):
                if data[i][j] == 1:
                    mu_err *= q_[j]
                    mu_not_err *= (1-q_[j])
                elif data[i][j] == 0:
                    mu_err *= (1-q_[j])
                    mu_not_err *= q_[j]
            mu_dirty[i] = mu_err
            mu_clean[i] = mu_not_err

        # M-step
        for j in range(len(data[0])):
            q_j = alpha - 1
            n_votes = 0.
            for i in range(len(data)):
                if data[i][j] == 0:
                    q_j += mu_clean[i]
                    n_votes += 1
                elif data[i][j] == 1:
                    q_j += mu_dirty[i]
                    n_votes += 1 
            q_j /= (n_votes + alpha + beta - 2)
            q_new[j] = q_j

    return np.sum(mu_dirty > mu_clean)
    
    
            

def remain_switch(data, pos_switch=True, neg_switch=True):
    data_subset = copy.deepcopy(data)
    majority = np.zeros((len(data_subset),len(data_subset[0])))
    switches = np.zeros((len(data_subset),len(data_subset[0])))
    for i in range(len(data_subset)):
        prev = 0
        for w in range(0,len(data_subset[0])):
            # the first worker is compared with an algorithmic worker
            n_w = np.sum(data[i][0:w+1] != -1)
            n_pos = np.sum(data[i][0:w+1] == 1)
            n_neg = np.sum(data[i][0:w+1] == 0)

            maj = 0
            if n_pos == n_neg and n_pos != 0:
                # tie results in switch
                maj = (prev + 1)%2
            elif n_pos > n_w/2:
                maj = 1
            if prev != maj:
                if (maj == 1 and pos_switch) or (maj == 0 and neg_switch):
                    switches[i][w] = 1
            prev = maj
            majority[i][w] = maj

    n_worker = np.sum(data_subset != -1, axis=1)
    n_all = n_worker
    data_subset[data_subset == -1] = 0

    histogram = n_worker
    n = float(np.sum(n_worker))
    n_bar = float(np.mean(n_worker))
    v_bar = float(np.var(n_worker))
    d = np.sum(np.logical_and(np.sum(switches,axis=1), n_all != 0)) 
    if n == 0:
        return d

    f1 = 0.
    for i in range(len(switches)):
        if n_worker[i] == 0:
            continue
        for k in range(len(switches[0])):
            j = len(switches[0]) -1 - k
            if data[i][j] == -1:
                continue
            elif switches[i][j] == 1:
                f1 += 1
            break

    # remove no-ops
    for i in range(len(switches)):
        switch_idx= np.where(switches[i,:]==1)[0]
        if len(switch_idx) > 0:
            n -= np.sum(data[i,:np.amin(switch_idx)] != -1)
        elif len(switch_idx) == 0:
            n -= np.sum(data[i,:] != -1)
    if n == 0:
        return d

    c_hat = max(1. - f1/n, 0.)
    if c_hat == 0.:
        return d

    gamma = v_bar/n_bar

    est = d/c_hat + n*(1-c_hat)/c_hat*gamma

    return est


def coeff_of_variance(hist):
    n = np.sum(hist)
    c = np.sum(hist > 0)
    f1 = float(np.sum(hist == 1))
    c_hat = 1. - f1/n
    
    s = 0.
    for i in range(2, len(hist)):
        s += np.sum(hist == i) * i * (i-1)
    gamma = s * (c/c_hat) / n / (n-1) - 1.
    
    return max(gamma,0)

def sample_coverage(data):
    hist = np.sum(data == 1, axis=1)
    n = float(np.sum(hist))
    f1 = float(np.sum(hist == 1))
    if n == 0:
        return 0.
    
    return 1. - f1/n

