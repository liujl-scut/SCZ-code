import multiprocessing
import time
import pandas as pd
import scipy.io as sio

from os import sep
from scipy import stats
from itertools import product
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from main_function import relevant_information, sturcture_dataset, set_record, set_record2, set_writer, check_path, \
    nested_cv

from main_run import run
from skrvm import RVR
from sklearn.svm import SVR
from sklearn_rvm import EMRVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor, ElasticNet

SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'rbans'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
cluster_merge = True  # [True, False]

model_name_select = ['MLPRegressor',
                     'KernelRidge',
                     'SGDRegressor',
                     'ElasticNet',]
model_select = [MLPRegressor(max_iter=1000),
                KernelRidge(),
                SGDRegressor(max_iter=10000, tol=1e-3),
                ElasticNet(random_state=0, alpha=0.15, l1_ratio=0.9), ]
param_grid_select = [{'learning_rate_init': [0.001, 0.01, 0.1, 0.2]},
                     {'alpha': [0.001, 0.01, 0.1, 0.5, 1]},
                     {},
                     {}]

if __name__ == '__main__':
    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select, False)
