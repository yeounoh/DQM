"""
    Data Quality Metric simulation/utility functions.
    
    (Developed in Python 2.7)
    
    Author: yeounoh chung (yeounohster@gmail.com)
"""
import numpy as np
import pylab as P
import matplotlib.pyplot as plt
import math, random, csv, pickle
#from estimator import *


"""
    Simulated datasets.
"""

def simulated_data(n_items=1000, n_workers=100, rho=0.2, w_coverage=0.02, w_precision=0.8, 
                   uniform_asgn=False):
    ''' 
        Take a random subset of data and assign it (task) to a random worker;
        this is embarassingly parallel and is used by SWITCH and VOTING 
        or most crowdsourced cleaning algorithms.
    '''

    # Prepare ground truth labels
    label = np.zeros(n_items)
    label[range(int(n_items*rho))] = 1
    random.shuffle(label)

    data = np.zeros((n_items, n_workers))-1
    for w in range(n_workers):
        items = np.random.choice(n_items, int(n_items*w_coverage))
        if uniform_asgn:
            # Uniformly assign items to workers (i.e., each item gets
            # the same number of workers).
            n_task = int(n_items*w_coverage)
            start = (w*n_task)%n_items
            end = ((w+1)*n_task)%n_items
            if end == 0:
                end = n_items
            items = range(start, end)
        
        for i in items:
            if random.random() <= w_precision:
                data[i,w] = label[i]
            else:
                data[i,w] = (label[i]+1)%2

    ground_truth = np.sum(label)

    return data, ground_truth

def simulation_with_triangular_walk(n_items=1000, n_workers=1000, 
                                    n_max=50, rho=0.02, w_coverage=0.02, w_precision=0.8):
    '''
        Simulate the sequential sampling procedure for triangular walks; we run n_items*w_coverage many
        parallel sequential sampling algorithms, and at each point we compute estimates for each of the
        algorithms (by default 20 of them), then average over them.
        Note that this is different from pooling all the linear estimates from the algorithms together and
        averaging.
        The latter turns out introduces a lot of statistical bias, whereas the current approach has negligible
        bias (at most 1%)
        For each of the algorithms, we keep running a single triangular walk until termination, put the linear
        estimate into the algorithm's bag, and sample a new item for a walk.
        As an anytime algorithm, we output the current estimate as the sample mean of all the estimates in that
        single bag.
    '''

    # Ground truth data generation
    label = np.zeros(n_items)
    label[range(int(n_items*rho))] = 1
    random.shuffle(label)

    batch_size = int(round(n_items*w_coverage))
    walks = {} # The current dict (keys: range(batch_size)) of batch_size many triangular walks in progress
    
    completed = set() # keeps track of completed triangles for batch item replacement
    estimates = dict() # estimates[n_worker_] = #error_estimate
    linear_estimates = dict() # One bag of linear estimates per algorithm, of which there are batch_size many

    # The first random batch of items and initialisation of linear estimate bags
    for i in range(batch_size):
        linear_estimates[i] = list()
        walks[i] = (np.random.choice(n_items), 0., 0.)

    # Triangular walks over the batch, with a total of n_workers/tasks
    n_workers_ = n_workers
    while n_workers_ > 0:
        for i in range(batch_size):
            item = walks[i][0]
            l_ = label[item]
            n_ = walks[i][1] + 1.
            k_ = walks[i][2]

            if l_ == 0 and random.random() > w_precision:
                k_ += 1.
            elif l_ == 1 and random.random() <= w_precision:
                k_ += 1.

            # check for stopping conditions
            if n_ < n_max and k_/n_ > 0.5:
                walks[i] = (walks[i][0], n_, k_)
            else:
                completed.add(i)
                if k_/n_ <= 0.5:
                    linear_estimates[i].append(0.)
                else:
                    if (2-n_max-2*k_)**2 -4*(2*n_max-2)*k_ >= 0:
                        p_ = ( 2.*k_+n_max-2+math.sqrt((2-n_max-2*k_)**2-4*(2*n_max-2)*k_)) / (4.*n_max-4)
                    else:
                        p_ = ((2.*k_+n_max-2)/(4*n_max-4))
                    #p_ = k_/n_max
                    #print 1./(2*p_-1.)
                    p_ = max(p_, 0.6)
                    linear_estimates[i].append(1./(2*p_-1.) - 0.165*p_*(1-p_)/(2*p_ - 1)**2)

        for i in completed:
            # reset because we allow sample with replacement and re-walk.
            walks[i] = (np.random.choice(n_items), 0., 0.)
            
        cur_estimates = list()

        for i in range(batch_size):
            if len(linear_estimates[i]) > 0:
                cur_estimates.append(np.mean(linear_estimates[i]))
        
        if len(cur_estimates) > 0:
            estimates[n_workers-n_workers_+1] = np.mean(cur_estimates) * n_items
                
        n_workers_ -= 1
        completed = set()
        
    return estimates
    

def simulation_with_parallel_triangular_walk(k, n_items=1000, n_workers=1000, 
                                    n_max=100, rho=0.02, w_coverage=0.02, w_precision=0.8):
    '''
        Simulate the parallel sampling procedure for triangular walks; that is, 
        we start with k random batches. The rest is similar to the sequential simulation.
    '''
    assert isinstance(k, (int, long))

    # Ground truth data generation
    label = np.zeros(n_items)
    label[range(int(n_items*rho))] = 1
    random.shuffle(label)
    
    data = {} # dataset with 
    for i in range(n_items):
        data[i] = (label[i], 0., 0.) # (label, n, k)
    
    completed = set() # keeps track of completed triangles for batch item replacement
    estimates = dict() # estimates[n_worker_] = #error_estimate
    linear_estimates = list()

    # k batches
    batches = []
    for i in range(k):
        items = list(np.random.choice(n_items, int(n_items*w_coverage), replace=True))
        batches.append(items)
       
    # Triangular walks over the k batches, with a total of n_workers/tasks
    # We need to keep track of tiangles indexed by item indices
    # so to aggregate all votes on the same item/triangle across multiple batches.
    n_workers_ = n_workers
    while n_workers_ > 0:
        batch_idx = random.randint(0,k-1)
        items = batches[batch_idx]
        for i in items:
            l_ = data[i][0]
            n_ = data[i][1] + 1.
            k_ = data[i][2] 

            if l_ == 0 and random.random() > w_precision:
                k_ += 1.
            elif l_ == 1 and random.random() <= w_precision:
                k_ += 1.

            # check for stopping conditions
            if n_ < n_max and k_/n_ > 0.5:
                data[i] = (l_, n_, k_)
            else:
                completed.add(i)
                if k_/n_ <= 0.5:
                    linear_estimates.append(0.)
                else:
                    if (2-n_max-2*k_)**2 -4*(2*n_max-2)*k_ >= 0:
                        p_ = ( 2.*k_+n_max-2+math.sqrt((2-n_max-2*k_)**2-4*(2*n_max-2)*k_)) / (4.*n_max-4)
                    else:
                        p_ = ((2.*k_+n_max-2)/(4*n_max-4))
                    linear_estimates.append(1./(2*p_-1.))

                # reset because we allow sample with replacement and re-walk.
                data[i] = (l_, 0., 0.)
            
        if len(linear_estimates) > 0:
            estimates[n_workers-n_workers_+1] = np.mean(linear_estimates) * n_items # #error_estimate

        # update batch (items): replace items with completed triangles
        # sample with replacement
        items = [i for i in items if i not in completed]
        items += list(np.random.choice(n_items, len(completed)))
        batches[batch_idx] = items
                
        n_workers_ -= 1
        completed = set()

    return estimates


def simulate_triangular_walk_by_triangles(n_items=1000, n_triangles=20000, 
                                    n_max=100, rho=0.02, w_coverage=0.02, w_precision=0.8, w_range=range(20000)):
    '''
        Simulate triangular walks measured by the number of tasks. 
    '''

    # Ground truth data generation
    label = np.zeros(n_items)
    label[range(int(n_items*rho))] = 1
    random.shuffle(label)
    
    walks = {} # dataset with 
               #walks[i] = (item_id, n, k, walk_id)
    
    completed = set() # keeps track of completed triangles for batch item replacement
    estimates = dict() # estimates[n_worker_] = #error_estimate
    results = dict() # for each walk_id, results[walk_id] is the output of the triangular walk with that id
    n_tasks = dict() # for each walk id, n_tasks[walk_id] stores the number of tasks it took to complete all triangles before and including walk_id
    id = 0

    # The first random batch of items
    #items = list(np.random.choice(n_items, int(n_items*w_coverage)))
    batch_size = int(round(n_items*w_coverage))
    for i in range(batch_size):
        walks[i] = (np.random.choice(n_items), 0., 0., id)
        id = id + 1

    # Triangular walks over the batch, with a total of n_workers/tasks
    n_workers_ = 0
    triangles_left = n_triangles
    while triangles_left > 0:
        n_workers_ = n_workers_ + 1
        for i in range(batch_size):
            item = walks[i][0]
            l_ = label[item]
            n_ = walks[i][1] + 1.
            k_ = walks[i][2]
            this_id = walks[i][3]

            if l_ == 0 and random.random() > w_precision:
                k_ += 1.
            elif l_ == 1 and random.random() <= w_precision:
                k_ += 1.

            # check for stopping conditions
            if n_ < n_max and k_/n_ > 0.5:
                walks[i] = (item, n_, k_, this_id)
            else:
                completed.add(i)
                if k_/n_ <= 0.5:
                    results[this_id] = 0.
                else:
                    if (2-n_max-2*k_)**2 -4*(2*n_max-2)*k_ >= 0:
                        p_ = ( 2.*k_+n_max-2+math.sqrt((2-n_max-2*k_)**2-4*(2*n_max-2)*k_)) / (4.*n_max-4)
                    else:
                        p_ = ((2.*k_+n_max-2)/(4*n_max-4))
                    results[this_id] = 1./(2*p_-1.)

        for i in completed:
            # reset because we allow sample with replacement and re-walk.
            n_tasks[walks[i][3]] = n_workers_ # only storing when this walk finished, need to take max later
            if walks[i][3] < n_triangles:
                triangles_left = triangles_left - 1
            walks[i] = (np.random.choice(n_items), 0., 0., id)
            id = id + 1
                
        completed = set()

    cur_n_tasks = 0
    cur_avg = 0
    for i in range(n_triangles):
        cur_n_tasks = max(n_tasks[i], cur_n_tasks)
        n_tasks[i] = cur_n_tasks
        cur_avg = cur_avg + (results[i] - cur_avg)/(i+1)
        estimates[i] = cur_avg * n_items

    return [estimates[w] for w in w_range], [n_tasks[w] for w in w_range]

"""
    Real-world datasets
"""
def restaurant_data(filename, priotization=True, wq_assurance=True):
    base_table = 'dataset/restaurant.csv'
    hard_pairs_ = pickle.load( open('dataset/hard_pairs.p','rb') )
    hard_pairs = {}

    with open("hard_pairs.txt", "w") as f:
        
        for p in hard_pairs_:
            f.write(str(p)+'\n')
            rid1 = int(p[0][0])
            rid2 = int(p[0][1])
            if rid1 < rid2:
                hard_pairs[(rid1,rid2)] = float(p[1])
            else:
                hard_pairs[(rid2,rid1)] = float(p[1])
    records = {}
    with open(base_table, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            gid = int(row[1]) # GT label
            rid = int(row[0]) # rid
            name = row[2]
            records[rid] = (gid, name)

    pair_solution = {}
    for rid1 in records.keys():
        for rid2 in records.keys():
            if rid1 == rid2:
                continue
            if priotization and (rid1,rid2) not in hard_pairs and (rid2,rid1) not in hard_pairs:
                    continue
            if (rid1,rid2) not in pair_solution and (rid2,rid1) not in pair_solution:
                if records[rid1][0] == records[rid2][0]:
                    pair_solution[(rid1,rid2)] = 1
                else:
                    pair_solution[(rid1,rid2)] = 0
    pickle.dump( pair_solution, open('dataset/rest_solution.p','wb') )
    pair_solution = pickle.load( open('dataset/rest_solution.p','rb') )

    easy_pair_solution = {}
    for rid1 in records.keys():
        for rid2 in records.keys():
            if rid1 == rid2:
                continue
            if (rid1,rid2) in hard_pairs or (rid2,rid1) in hard_pairs:
                continue
            if (rid1,rid2) not in easy_pair_solution and (rid2,rid1) not in easy_pair_solution:
                if records[rid1][0] == records[rid2][0]:
                    easy_pair_solution[(rid1,rid2)] = 1
                else:
                    easy_pair_solution[(rid1,rid2)] = 0
    pickle.dump( easy_pair_solution, open('dataset/easy_rest_solution.p','wb') )
    easy_pair_solution = pickle.load( open('dataset/easy_rest_solution.p','rb') )


    task_resp = {}
    worker_resp = {}
    for asgn_table in filename:
        with open(asgn_table,'rb') as f:
            reader = csv.reader(f)
            for asgn in reader:
                if asgn[0] == 'id':
                    continue
                w = asgn[4] #asgn[3]
                task = asgn[2]

                answers = asgn[1][1:-2].replace("\"","").replace("Pair","").split(",")
                for ans in answers:
                    rids = ans.split(":")[0].strip().split("-")
                    resp = float(ans.split(":")[1])
                    sim = jaccard(records[int(rids[0])][1],records[int(rids[1])][1])
                    tup = ( (int(rids[0]),int(rids[1])), resp )
                    if tup[0] not in pair_solution:
                        tup = ( (int(rids[1]),int(rids[0])), resp )
                    if priotization and tup[0] not in hard_pairs:
                        continue
                    # for worker_resp
                    if w in worker_resp:
                        if tup in worker_resp[w]:
                            continue
                        else:
                            worker_resp[w].append(tup)
                    else:
                        worker_resp[w] = [tup]
                    # for task_resp
                    if task in task_resp:
                        if tup in task_resp[task]:
                            continue
                        else:
                            task_resp[task].append(tup)
                    else:
                        task_resp[task] = [tup]
        print ('#workers:',len(worker_resp), '#tasks:',len(task_resp))
    print ('worker_resp loaded')

    # worker evaluation
    score = []
    for w in worker_resp.keys():
        correct = 0.
        for res in worker_resp[w]:
            if res[0][0] < res[0][1]:
                if pair_solution[res[0]] == res[1]:
                    correct += 1.
            else:
                if pair_solution[(res[0][1],res[0][0])] == res[1]:
                    correct += 1.
        score.append(correct/len(worker_resp[w]))
        # remove bad workers
        if wq_assurance and score[-1] < 0.6:
            worker_resp.pop(w)
    score = np.array(score)
    print ('#bad workers:',np.sum(score < 0.6))

    ilist_workers = worker_resp.keys()
    ilist_tasks = task_resp.keys()
    ilist_pairs = pair_solution.keys()
    lookup_tbl = {}
    for i in range(len(ilist_pairs)):
        lookup_tbl[ilist_pairs[i]] = i

    data = np.zeros((len(ilist_pairs),len(ilist_workers))) + -1
    print ('#pairs: ', len(ilist_pairs), '#workers: ', len(ilist_workers))
    
    
    for k,v in worker_resp.iteritems():
    #for k,v in task_resp.iteritems():
        for pair_resp in v:
            if pair_resp[0][0] < pair_resp[0][1]:
                data[lookup_tbl[pair_resp[0]],ilist_workers.index(k)] = pair_resp[1]
            else:
                data[lookup_tbl[(pair_resp[0][1],pair_resp[0][0])],ilist_workers.index(k)] = pair_resp[1]

    return data, np.sum(pair_solution.values())    


"""
    Utility functions
"""

def holdout_workers(bdataset, gt_list, worker_range, est_list, 
                    rel_err=False, rep=1):
    X = []
    Y = []
    GT = []

    for w in worker_range:
        random_trial = np.zeros((len(est_list)+len(gt_list),rep))
        for t in range(0,rep):
            dataset = bdataset[:,
                               np.random.choice(range(len(bdataset[0])),
                                                min(w,len(bdataset[0])),replace=False)]
            for e in est_list:
                # ground truth ( len(est_list) == len(gt_list) )
                A = gt_list[est_list.index(e)]
                if callable(A):
                    A = float(gt_list[est_list.index(e)](dataset))
                else:
                    A = float(A)
                random_trial[len(est_list) + est_list.index(e),t] = A

                if rel_err: #SRMSE
                    random_trial[est_list.index(e),t] = (e(dataset)-A)**2 / A**2
                else:
                    random_trial[est_list.index(e),t] = e(dataset)

        result_array = []
        var_array = []
        if rel_err:
            result_array = np.sqrt(np.mean(random_trial[0:len(est_list),:],axis=1))
            var_array = np.std(random_trial[0:len(est_list),:],axis=1)
        else:
            result_array = np.mean(random_trial[0:len(est_list),:],axis=1)
            var_array = np.std(random_trial[0:len(est_list),:],axis=1)

        gt_array = np.mean(random_trial[len(est_list):len(est_list)+len(gt_list),:],axis=1)
        if rel_err:
            gt_array = []

        X.append(w)
        Y.append( [list(result_array),list(var_array)] )
        GT.append(list(gt_array))

    return (X,Y,GT)

def plotY1Y2(points,
             title="",
             xaxis="",
             yaxis="Estimate",
             ymax=-1,
             xmin=-1,
             legend=[],
             legend_gt=[],
             loc = 'upper right',
             filename="output.png",
             logscale=False,
             rel_err=False,
             font=20):
    from matplotlib import font_manager, rcParams
    rcParams.update({'figure.autolayout': True})
    rcParams.update({'font.size': font})
    fprop = font_manager.FontProperties(fname='/Library/Fonts/Microsoft/Gill Sans MT.ttf')

    num_estimators = len(legend)#len(points[1][0][0])
    num_gt = len(legend_gt)#len(points[2][0])
    if rel_err:
        num_gt=0

    fig, ax = plt.subplots(1,figsize=(8,5))
    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    markers = ['o-','v-','^-','s-','*-','x-','+-','D-']
    shapes = ['--','-*']

    for i in range(num_gt):
        res = [j[i] for j in points[2]]
        if logscale:
            ax.semilogy(points[0], res, shapes[i],linewidth=2.5,color="#333333",label=legend_gt[i])
        else:
            ax.plot(points[0], res, shapes[i],linewidth=2.5,color="#333333",label=legend_gt[i])

    for i in range(num_estimators):
        #if i is not 0: continue # to plot EXTRAPOL only
        res = np.array([j[0][i] for j in points[1]])
        if not rel_err:
            std = np.array([j[1][i] for j in points[1]])

        if logscale:
            plt.semilogy(points[0], res, 's-', linewidth=2.5, markersize=7, color=colors[i], label=legend_gt[i])
        else:
            if not rel_err and 'EXTRAPOL' in legend[i]:
                #ax.plot(points[0], np.zeros(len(points[0]))+res[0], '--',linewidth=1.5, color='g',label=legend[i]) 
                ax.fill_between(points[0], res[0]-1*std[0], res[0]+1*std[0], 
                                alpha=0.3, edgecolor='#CC4F1B', facecolor='#FF9848')#,label=legend[i])
                #ax.errorbar(points[0],np.zeros(len(points[0]))+res[0],yerr=std[0],linewidth=2.5,color='g',label=legend[i])
            else:
                if rel_err:
                    ax.plot(points[0], res, markers[i], 
                            linewidth=2.5,markersize=7,color=colors[i],label=legend[i])
                else:
                    ax.errorbar(points[0], res, yerr=std, fmt=markers[i], 
                                linewidth=2,markersize=7,color=colors[i],label=legend[i])
                #for z in range(len(points[0])):
                #    print legend[i], points[0][z], res[z]

    ax.set_title(title,fontsize=font)
    ax.set_xlabel(xaxis,fontsize=font)#,fontproperties=fprop)
    ax.set_ylabel(yaxis,fontsize=font)#,fontproperties=fprop)
    if not logscale and ymax == -1:
        ax.set_ylim(ymin=0)
    elif not logscale:
        ax.set_ylim(ymin=0,ymax=ymax)

    if xmin == -1:
        ax.set_xlim(xmin=points[0][0], xmax=points[0][len(points[0])-1])
    else:
        ax.set_xlim(xmin=xmin, xmax=points[0][len(points[0])-1])

    ax.legend(loc=loc,prop={'size':15}).get_frame().set_alpha(0.5)
    ax.grid()
    fig.savefig(filename,bbox_inches='tight')#,format='pdf')

def jaccard(a,b):
    word_set_a = set(a.lower().split())
    word_set_b = set(b.lower().split())
    word_set_c = word_set_a.intersection(word_set_b)
    return float(len(word_set_c)) / (len(word_set_a) + len(word_set_b) - len(word_set_c))


if __name__=="__main__":
        data, gt = restaurant_data(['dataset/good_worker/restaurant_additional.csv',
                                    'dataset/good_worker/restaurant_new.csv'])
        pair_solution = pickle.load( open('dataset/rest_solution.p','rb') )
        slist = pair_solution.values()
        easy_pair_solution = pickle.load( open('dataset/easy_rest_solution.p','rb') ) 
        easy_slist = easy_pair_solution.values()
        obvious_err = np.sum(np.array(easy_slist) == 1) 
