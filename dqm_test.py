import unittest
from data_util import *
from estimator import *
from time import time
import pickle
import numpy as np

class DQMTest(unittest.TestCase):
    
    def test_estimators(self):
        print ('<<< Estimators Test >>>')
        est_list = [#lambda x: nominal(x) + obvious_err, 
                    lambda x: vNominal(x) + obvious_err, 
                    #lambda x: switch(x) + obvious_err,
                    #lambda x: expectation_maximization(x, alpha=0.7, beta=0.2) + obvious_err,
                    #lambda x: triangular_walk(x,n_max=5) + obvious_err,
                    lambda x: triangular_walk(x,n_max=10) + obvious_err,
                    #lambda x: triangular_walk(x,n_max=30) + obvious_err,
                   ]
                    
        gt_list = [#lambda x: gt + obvious_err, 
                   lambda x: gt + obvious_err, 
                   #lambda x: gt + obvious_err, 
                   #lambda x: gt + obvious_err,
                   lambda x: gt + obvious_err,
                   #lambda x: gt + obvious_err,
                  ]
        legend = ["VOTING", "T-WALK (10)"]
        legend_gt = ["Ground Truth"]

        # Restaurant data
        '''
        data, gt = restaurant_data(['dataset/good_worker/restaurant_additional.csv',
                                    'dataset/good_worker/restaurant_new.csv'])
        pair_solution = pickle.load( open('dataset/rest_solution.p','rb') )
        slist = pair_solution.values()
        easy_pair_solution = pickle.load( open('dataset/easy_rest_solution.p','rb') ) 
        easy_slist = easy_pair_solution.values()
        obvious_err = np.sum(np.array(easy_slist) == 1)
        (X, Y, GT) = holdout_workers(data, gt_list, range(100, 1400, 100), est_list, rep=3)
        plotY1Y2((X,Y,GT), legend=legend, legend_gt=legend_gt,
                 xaxis='Tasks', yaxis='# Error Estimate',
                 xmin=200, loc='lower right', title='Restaurant Data', 
                 filename='figure/test_restaurant_data.png')
        '''
        # Simulated data
        ## Worker parameters
        n_workers = 1400
        w_number = range(200, n_workers, 200)
        w_quality = [0.7, 0.8, 0.95] 
        w_c = 0.02
        ## Error proportion
        n_items = 1000
        rho = 0.02
        obvious_err = 0

        for w_q in w_quality:
            data, gt = simulated_data(n_items, n_workers, rho, w_coverage=w_c, w_precision=w_q)

            (X, Y, GT) = holdout_workers(data, gt_list, w_number, est_list, rep=100)
            plotY1Y2((X,Y,GT), legend=legend, legend_gt=legend_gt,
                     xaxis='Tasks', yaxis='# Errors',
                     ymax=50, xmin=200, loc='best', title='Worker quality: %s'%str(w_q),
                     filename='figure/test_estimators_r%s_c%s_q%s.png'%(rho,w_c,w_q))




    def test_different_n_max_values(self):
        n_items = 1000
        init = 20
        n_workers = 400
        step = 20
        w_range = range(init, n_workers, step)
        n_rep = 5

        est_list = [vNominal]
        gt_list = [lambda x: gt]
        legend = ["T-WALK (3)", "T-WALK (4)", "T-WALK (5)",
                    "T-WALK(6)", "T-WALK(10)","T-WALK(20)","T-WALK(50)"]
        legend_gt = ["Ground Truth"]
        
        for rho in [0.01, 0.05, 0.1]:
            for cov in [20./n_items]:
                data, gt = simulated_data(n_items, n_workers, rho, w_coverage=cov)
                (X_, Y_, GT_) = holdout_workers(data, gt_list, w_range, est_list, rep=n_rep)
                voting_results = {}
                #switch_results = {}
                for i in range(len(w_range)):
                    voting_results[w_range[i]] = (Y_[i][0][0], Y_[i][1][0])
                    #switch_results[w_range[i]] = (Y_[i][0][1], Y_[i][1][1])

                n_max_ = [3,4,5,6,10,20,50]
                avg_ = {}
                std_ = {}
                for n_max in n_max_:
                    est_ = []
                    for i in range(n_rep):
                        est_dict = simulation_with_triangular_walk(n_items=n_items, rho=rho, n_workers=n_workers, 
                                                                   n_max=n_max, w_coverage=cov)
                        est_.append([est_dict[w] for w in w_range])
                    avg_[n_max] = np.mean(est_, axis=0)
                    std_[n_max] = np.std(est_, axis=0)

                X, Y, GT = [], [], []
                for i in range(len(w_range)):
                    X.append(w_range[i])
                    Y.append( [ [#voting_results[w_range[i]][0], 
                                 #switch_results[w_range[i]][0],
                                 avg_[3][i], avg_[4][i], avg_[5][i],
                                 avg_[6][i], avg_[10][i],
                                 avg_[20][i], avg_[50][i]], 
                                [#voting_results[w_range[i]][1], 
                                 #switch_results[w_range[i]][1],
                                 std_[3][i], std_[4][i], std_[5][i],
                                 std_[6][i], std_[10][i],
                                 std_[20][i], std_[50][i]] ] )
                    GT.append([gt])

                plotY1Y2((X,Y,GT), legend=legend, legend_gt=legend_gt,
                         xaxis='Tasks', yaxis='# Error Estimate', 
                         #ymax=n_items*rho*1.8, 
                         xmin=init, loc='best', title='Batch size: %s x %s, rho: %s'%(n_items,cov,rho),
                         filename='figure/test_simulated_with_tri_walk_c%s_r%s.png'%(cov,rho))


    def test_robustness_to_fp_fn(self):
        n_items = 1000
        init = 20
        n_workers = 400
        step = 20
        w_range = range(init, n_workers, step)
        n_rep = 5
        fnr, fpr = 0.0, 0.01

        est_list = [vNominal, switch]
        gt_list = [lambda x: gt, lambda x: gt]
        legend = ["VOTING", "SWITCH", "T-WALK(10)","T-WALK(20)","T-WALK(30)"]
        legend_gt = ["Ground Truth"]
        
        for rho in [0.01, 0.05]:
            for cov in [20./n_items]:
                data, gt = simulated_data2(n_items, n_workers, rho, w_coverage=cov, fnr=fnr, fpr=fpr)
                (X_, Y_, GT_) = holdout_workers(data, gt_list, w_range, est_list, rep=n_rep)
                voting_results = {}
                switch_results = {}
                for i in range(len(w_range)):
                    voting_results[w_range[i]] = (Y_[i][0][0], Y_[i][1][0])
                    switch_results[w_range[i]] = (Y_[i][0][1], Y_[i][1][1])

                n_max_ = [10,20,30]
                avg_ = {}
                std_ = {}
                for n_max in n_max_:
                    est_ = []
                    for i in range(n_rep):
                        est_dict = simulation_with_triangular_walk(n_items=n_items, rho=rho, n_workers=n_workers, 
                                                                   n_max=n_max, w_coverage=cov)
                        est_.append([est_dict[w] for w in w_range])
                    avg_[n_max] = np.mean(est_, axis=0)
                    std_[n_max] = np.std(est_, axis=0)

                X, Y, GT = [], [], []
                for i in range(len(w_range)):
                    X.append(w_range[i])
                    Y.append( [ [voting_results[w_range[i]][0], 
                                 switch_results[w_range[i]][0],
                                 avg_[10][i], avg_[20][i], avg_[30][i]],
                                [voting_results[w_range[i]][1], 
                                 switch_results[w_range[i]][1],
                                 std_[10][i], std_[20][i], std_[30][i]] ] )
                    GT.append([gt])

                plotY1Y2((X,Y,GT), legend=legend, legend_gt=legend_gt,
                         xaxis='Tasks', yaxis='# Error Estimate', 
                         #ymax=n_items*rho*1.8, 
                         xmin=init, loc='best', title='Batch size:%sx%s, rho:%s, fnr:%s, fpr:%s'%(n_items,cov,rho,fnr,fpr),
                         filename='figure/test_simulated_with_tri_walk_c%s_r%s_fnr%s_fpr%s.png'%(cov,rho,fnr,fpr))

if __name__ == '__main__':
    unittest.main()
