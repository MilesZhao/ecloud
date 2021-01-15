import numpy as np
import pandas as pd
seed = 123
np.random.seed(seed)
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymatgen.core.composition import Composition
from util import *
from collections import Counter


_,y,names=load_all(typ = 'bulk')

names = [ s.split("_")[1] for s in names]
df_formulas = pd.DataFrame(
                names, 
                columns = ['formulas']
            )
df = StrToComposition(target_col_id='composition').featurize_dataframe(df_formulas, 'formulas')
magpie_feat = ElementProperty.from_preset(preset_name='magpie')
magpie_df = magpie_feat.featurize_dataframe(df, col_id="composition",ignore_errors=False)

fts =  magpie_df.iloc[:,2:].values
y = y.reshape(-1,1)
dat = np.concatenate((y, fts), axis=1)
y = dat[:,0]
X = dat[:,1:]


kf = KFold(n_splits=5, shuffle=True)
rf = RandomForestRegressor(n_estimators=50)

r2_scores, mse_scores, rmse_scores, rc_scores = [], [], [], []
for train_index, val_index in kf.split(X):
    trn_x, trn_y = X[train_index], y[train_index]
    val_x, val_y = X[val_index], y[val_index]
    rf.fit(trn_x, trn_y)
    preds = rf.predict(val_x)
    r2, mse, rmse, rc = eval_metrics(val_y, preds)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    rc_scores.append(rc)


print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (5, np.mean(r2_scores)))
print('Folds: %i, mean MAE: %.3f' % (5, np.mean(mse_scores)))
print('Folds: %i, mean RMSE: %.3f' % (5, np.mean(rmse_scores)))
print('Folds: %i, mean Rank Coef: %.3f' % (5, np.mean(rc_scores)))
exit()




















