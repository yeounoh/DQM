import unittest
from data_util import *
from estimator import *
from time import time
import pickle
import numpy as np

class test_estimator(unittest.TestCase):
    
    def test_estimators(self):
        print ('<<< Estimators Test >>>')
        est_list = [#lambda x: nominal(x) + obvious_err, 
                    lambda x: vNominal(x) + obvious_err, 
                    #lambda x: switch(x) + obvious_err,
                    lambda x: triangular_walk(x,n_max=3) + obvious_err,
                    lambda x: triangular_walk(x,n_max=5) + obvious_err,
                    lambda x: triangular_walk(x,n_max=10) + obvious_err,
                    lambda x: triangular_walk(x,n_max=15) + obvious_err]
                    
        gt_list = [lambda x: gt + obvious_err, 
                   lambda x: gt + obvious_err, 
                   lambda x: gt + obvious_err,
                   lambda x: gt + obvious_err,
                   lambda x: gt + obvious_err]
        legend = ["VOTING", "T-WALK (3)", "T-WALK (5)", "T-WALK (10)", "T-WALK (15)"]
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
        data, gt = simulated_data(1000,800,0.5,uniform_asgn=False)
        obvious_err = 0
        (X, Y, GT) = holdout_workers(data, gt_list, range(50,850,50), est_list, rep=3)
        plotY1Y2((X,Y,GT), legend=legend, legend_gt=legend_gt,
                 xaxis='Tasks', yaxis='# Error Estimate',
                 ymax=1000, xmin=100, loc='best', title='Simulated Data',
                 filename='figure/test_simulated_data.png')

        data, gt = simulated_data(1000,500,0.5,uniform_asgn=True)
        obvious_err = 0
        (X, Y, GT) = holdout_workers(data, gt_list, range(50,550,50), est_list, rep=3)
        plotY1Y2((X,Y,GT), legend=legend, legend_gt=legend_gt,
                 xaxis='Tasks', yaxis='# Error Estimate',
                 ymax=1000, xmin=100, loc='best', title='Simulated Data',
                 filename='figure/test_simulated_data_uniform_worker_asgn.png')

    def test_simulation_with_triangular_walk(self):
        data, gt = simulated_data(1000, 600, 0.5)
        w_range = range(50, 600, 50)
        n_rep = 5

        est_list = [vNominal] 
        gt_list = [lambda x: gt]
        legend = ["VOTING", "T-WALK (3)", "T-WALK (10)", "T-WALK (30)"]
        legend_gt = ["Ground Truth"]
        (X_, Y_, GT_) = holdout_workers(data, gt_list, w_range, est_list, rep=n_rep)
        est_results = {}
        for i in range(len(w_range)):
            est_results[w_range[i]] = (Y_[i][0][0], Y_[i][1][0])
        
        for cov in [0.001, 0.02]:

            n_max_ = [3, 10, 30]
            avg_ = {}
            std_ = {}
            for n_max in n_max_:
                est_ = []
                for i in range(n_rep):
                    est_dict = simulation_with_triangular_walk(rho=0.5, n_workers=600, n_max=n_max, w_coverage=cov)
                    est_.append([est_dict[w] for w in w_range])
                avg_[n_max] = np.mean(est_, axis=0)
                std_[n_max] = np.std(est_, axis=0)

            X, Y, GT = [], [], []
            for i in range(len(w_range)):
                X.append(w_range[i])
                Y.append( [ [est_results[w_range[i]][0], avg_[3][i], avg_[10][i], avg_[30][i]], 
                            [est_results[w_range[i]][1], std_[3][i], std_[10][i], std_[30][i]] ] )
                GT.append([gt])
            
            plotY1Y2((X,Y,GT), legend=legend, legend_gt=legend_gt,
                     xaxis='Tasks', yaxis='# Error Estimate',
                     ymax=1000, xmin=100, loc='best', title='Batch size: 1000x%s'%cov,
                     filename='figure/test_simulated_with_tri_walk_c%s.png'%cov)

if __name__ == '__main__':
    unittest.main()
