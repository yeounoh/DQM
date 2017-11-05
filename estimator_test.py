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
                    lambda x: triangular_walk(x,n_max=7) + obvious_err,
                    lambda x: triangular_walk(x,n_max=9) + obvious_err,
                    lambda x: triangular_walk(x,n_max=11) + obvious_err]
                    
        gt_list = [lambda x: gt + obvious_err, 
                   lambda x: gt + obvious_err, 
                   lambda x: gt + obvious_err,
                   lambda x: gt + obvious_err,
                   lambda x: gt + obvious_err,
                   lambda x: gt + obvious_err]
        legend = ["VOTING", "T-WALK (3)", "T-WALK (5)", "T-WALK (7)", "T-WALK (9)", "T-WALK (11)"]
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
        data, gt = simulated_data(1000,500,0.5,uniform_asgn=False)
        obvious_err = 0
        (X, Y, GT) = holdout_workers(data, gt_list, range(50,550,50), est_list, rep=3)
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

if __name__ == '__main__':
    unittest.main()
