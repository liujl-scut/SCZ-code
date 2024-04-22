from main_run import run
from sklearn.linear_model import SGDRegressor


SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'rbans'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['merge']
cluster_merge = True  # [True, False]


def model_sgd(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['SGDRegressor',]
    model_select = [SGDRegressor(max_iter=10000, tol=1e-3)]
    param_grid_select = [{'alpha': [0.001, 0.01, 0.1, 0.5, 1]},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_sgd(SCZ, scale, feature, band, cluster_merge)
