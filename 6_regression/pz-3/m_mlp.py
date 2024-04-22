from main_run import run
from sklearn.neural_network import MLPRegressor


SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
cluster_merge = False  # [True, False]

def model_mlp(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['MLPRegressor']
    model_select = [MLPRegressor(max_iter=10000),]
    param_grid_select = [{'n_estimators': [30, 50, 100]},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_mlp(SCZ, scale, feature, band, cluster_merge)