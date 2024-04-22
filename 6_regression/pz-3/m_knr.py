from main_run import run
from sklearn.neighbors import KNeighborsRegressor


SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
cluster_merge = False  # [True, False]


def model_knr(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['KNeighborsRegressor',]
    model_select = [KNeighborsRegressor(n_jobs=-1),]
    param_grid_select = [{'n_neighbors': [3, 4, 5, 6, 7, 8]},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_knr(SCZ, scale, feature, band, cluster_merge)
