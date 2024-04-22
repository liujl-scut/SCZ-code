from main_run import run
from sklearn.linear_model import Lasso


SCZ = 'ningbo'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['pec', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['ALPHA', 'BETA']
cluster_merge = False  # [True, False]


def model_lasso(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['Lasso']
    model_select = [Lasso(max_iter=10000),]
    param_grid_select = [{'alpha': [0.01, 0.1, 1.0, 2, 3]},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_lasso(SCZ, scale, feature, band, cluster_merge)
