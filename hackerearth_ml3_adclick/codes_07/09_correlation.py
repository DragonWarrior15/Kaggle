import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
from scipy.stats import pearsonr
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import sklearn.metrics as skmetrics

# with open(c_vars.train_spilt_val_processed, 'rb') as f:
    # X, y = pickle.load(f)

with open('../analysis_graphs/standard_scaler', 'rb') as f:
    standard_scaler = pickle.load(f)

# with open(c_vars.train_spilt_train_processed, 'rb') as f:
with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_unseen, y_unseen = pickle.load(f)

country_codes = np.array(standard_scaler.inverse_transform(X_unseen)[:,0]).reshape(-1,1).astype(np.int64)

indices_array_unseen = [0 for _ in range(2)]
indices_array_unseen[0] = np.array(list(country_codes == 1)).reshape(-1)
indices_array_unseen[1] = np.array(list(country_codes != 1)).reshape(-1)

X_unseen = [X_unseen[idx,:] for idx in indices_array_unseen]
y_unseen = [y_unseen[idx,] for idx in indices_array_unseen]

# for i in range(X.shape[1] - 1):
    # for j in range(i + 1, X.shape[1]):
        # print (str(i) + ',' + str(j) + ',' + str(pearsonr(X[:,i], X[:,j])))
# print ('#####')

for X, y in zip(X_unseen, y_unseen):
    print ('##')
    for i in range(X.shape[1]):
        print (str(i) + ',' + str(pearsonr(X[:,i], y)))