from main_run import run
from sklearn.linear_model import LinearRegression


SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'rbans'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
cluster_merge = False  # [True, False]


def model_lr(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['LinearRegression',]
    model_select = [LinearRegression(normalize=True, n_jobs=-1),]
    param_grid_select = [{},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_lr(SCZ, scale, feature, band, cluster_merge)
