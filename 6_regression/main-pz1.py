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
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']  # 少一个all
cluster_merge = False  # [True, False]

model_name_select = ['RandomForestRegressor',]
model_select = [RandomForestRegressor(n_jobs=-1)]
param_grid_select = [{}]

if __name__ == '__main__':
    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select, False)
