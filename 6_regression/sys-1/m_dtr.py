from main_run import run
from sklearn.tree import DecisionTreeRegressor


SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'rbans'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['pec', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['ALPHA', 'BETA']
cluster_merge = False  # [True, False]


def model_dtr(SCZ, scale, feature, band, cluster_merge):
    model_name_select = ['DecisionTreeRegressor']
    model_select = [DecisionTreeRegressor(),]
    param_grid_select = [{},]

    run(SCZ, scale, feature, band, cluster_merge, model_name_select, model_select, param_grid_select)


model_dtr(SCZ, scale, feature, band, cluster_merge)
