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

# from xgboost.xgb import XGBRegressor
# from lightgbm.lgb import LGBMRegressor
# 'LGBMRegressor',
# 'XGBRegressor'
# LGBMRegressor(num_leaves=31, learning_rate=0.5, n_estimators=200, n_jobs=-1),
# XGBRegressor(max_depth=2, learning_rate=0.001, n_estimators=300, objective='reg:squarederror',
#              n_jobs=-1)
# {},
# {}

SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['pec', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['ALPHA', 'BETA']
cluster_merge = False  # [True, False]

model_name_select = ['SVR',
                     'Lasso',
                     'RVR',
                     'LinearRegression',
                     'KNeighborsRegressor',
                     'DecisionTreeRegressor',
                     'RandomForestRegressor',
                     'MLPRegressor',
                     'KernelRidge',
                     'SGDRegressor',
                     'ElasticNet', ]
model_select = [SVR(kernel='linear'),
                Lasso(max_iter=10000),
                RVR(kernel='linear', n_iter=10000),
                LinearRegression(normalize=True, n_jobs=-1),
                KNeighborsRegressor(n_jobs=-1),
                DecisionTreeRegressor(),
                RandomForestRegressor(n_jobs=-1),
                MLPRegressor(max_iter=1000),
                KernelRidge(),
                SGDRegressor(max_iter=10000, tol=1e-3),
                ElasticNet(random_state=0, alpha=0.15, l1_ratio=0.9), ]
param_grid_select = [{'C': [0.0001, 0.001, 0.01, 0.1]},
                     {'alpha': [0.01, 0.1, 1.0, 2, 3]},
                     {},
                     {},
                     {'n_neighbors': [3, 4, 5, 6, 7, 8]},
                     {},
                     {},
                     {'learning_rate_init': [0.001, 0.01, 0.1, 0.2]},
                     {'alpha': [0.001, 0.01, 0.1, 0.5, 1]},
                     {}, ]

if __name__ == '__main__':
    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select, False)
