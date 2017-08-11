import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold, cross_val_score
import sklearn.metrics as skmetrics

from sklearn.ensemble import GradientBoostingClassifier

# np.random.seed(42)
train_columns = [i for i in range(73)]
# train_columns = [64,56,23,67,48,44,59,28,60,52,62,63,16,27,51]
# train_columns = [64,56,67,44,59,28,62,16,51]
# train_columns = [64,60,24,28,68,23,27,56,44,22,52,21,48,26,25,47,45,46,12,9,10,11]
print (train_columns)

print(str(datetime.now()) + ' Reading Data')
with open(c_vars.train_spilt_train_processed, 'rb') as f:
    X, y = pickle.load(f)

with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_unseen, y_unseen = pickle.load(f)

print(str(datetime.now()) + ' Reading Data Complete')

# param_dict = {'learning_rate' : [0.05, 1, 3],
#               'max_depth' : [5, 10, 50, 100, 200],
#               'n_estimators' : [50, 100, 200]}

param_dict = {'max_depth' : [4],
              'n_estimators' : [10],
              'learning_rate' : [1]
              }

model_list = []

param_space, param_to_int_dict = c_vars.get_param_space(param_dict)
# print (param_space)
# print (param_to_int_dict)
param_space = [[1, 3, 20]]
# param_list = [0.05, 5, 200]
for param_list in param_space:
    print(str(datetime.now()) + ' Training Gradient Boosted classifier, ' + str(param_list))
    kf = KFold(n_splits = 4, shuffle = True)
    clf = GradientBoostingClassifier(max_depth        = param_list[param_to_int_dict['max_depth']],
                                     n_estimators     = param_list[param_to_int_dict['n_estimators']],
                                     learning_rate    = param_list[param_to_int_dict['learning_rate']],
                                     random_state = 42
                                    )
    kf_index = 0
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        X_train, X_test = X_train[:, train_columns], X_test[:, train_columns]
        y_train, y_test = y[train_indices], y[test_indices]

        clf.fit(X_train, y_train)
        model_list.append(clf)
        print (str(datetime.now()) + ' Model Training Complete')

        for myX, myY, Set in zip([X_train, X_test, X_unseen], [y_train, y_test, y_unseen], ['Train', 'Test', 'Unseen']):
            y_pred = clf.predict(myX)
            y_pred_proba = clf.predict_proba(myX)

            print (str(datetime.now()) + ' KF_Index,' + Set + ',Accuracy,Confusion_Matrix,AUC')
            print (','.join([str(kf_index), Set, str(skmetrics.accuracy_score(myY, y_pred)), 
                        str(skmetrics.confusion_matrix(myY, y_pred).tolist()), str(skmetrics.roc_auc_score(myY, y_pred_proba[:,1]))]))

        # print (clf.feature_importances_)
        kf_index += 1

with open('../analysis_graphs/gbc_20170809_1157', 'wb') as f:
    pickle.dump(model_list, f)

# rf_20170809_1212 0.63508 LB
# rf_20170809_1622 0.67025 LB, only using first 15 columns of train
# rf_20170809_1800 0.64919 LB, only using first 33 columns of train
# lr_20170809_1622 0.67096 LB, using [64, 60, 24, 28, 68, 23, 27, 56, 44, 22, 52, 21, 48, 26, 25, 47, 45, 46, 12, 9, 10, 11] of train
# xgb_20170809_1157 0.59261 LB, using [64,56,23,67,48,44,59,28,60,52,62,63,16,27,51]
'''
with open('../analysis_graphs/gbc_20170809_1157', 'rb') as f:
    model_list = pickle.load(f)


print (str(datetime.now()) + ' Reading submit set')
# df_id = pd.read_csv(c_vars.test_processed_id, usecols = ['ID'])
df_id = pd.read_csv(c_vars.test_processed_id)
with open(c_vars.test_processed, 'rb') as f:
  X_submit = pickle.load(f)
print (str(datetime.now()) + ' Predicting on submit set')
y_submit_pred_proba_1 = np.sum([0.25 * model_list[i].predict_proba(X_submit[:, train_columns])[:,1] for i in range(4)], axis = 0)
# y_submit_pred_proba_1 = model_list[3].predict_proba(X_submit[:, train_columns])[:,1]
df_id['click'] = y_submit_pred_proba_1

df_id[['ID', 'click']].to_csv('../output/submit_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv', index = False)
print (str(datetime.now()) + ' Done')
'''