from main_run import run
from sklearn.ensemble import RandomForestRegressor


SCZ = 'ningbo'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['pec', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['ALPHA', 'BETA']
cluster_merge = False  # [True, False]


def model_rfr(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['RandomForestRegressor',]
    model_select = [RandomForestRegressor(n_jobs=-1),]
    param_grid_select = [{'n_estimators': [30, 50, 100]},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_rfr(SCZ, scale, feature, band, cluster_merge)
