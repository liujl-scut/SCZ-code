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
band = ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
cluster_merge = False  # [True, False]

model_name_select = ['MLPRegressor',]
model_select = [MLPRegressor(max_iter=1000),]
param_grid_select = [{'learning_rate_init': [0.001, 0.01, 0.1, 0.2]}]

if __name__ == '__main__':
    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select, False)
