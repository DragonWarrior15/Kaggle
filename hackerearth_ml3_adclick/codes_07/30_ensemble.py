import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
from scipy.stats import pearsonr, mode
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import KFold, cross_val_score
import sklearn.metrics as skmetrics

model_column_dict = {
                     'rf_20170813_1800'  : [85,93,105,109,25,108,81,77,29,37],
                     # 'rf_20170813_1830'  : [i for i in range(130)],
                     'lr_20170813_1800'  : [89,117,73,81,101,116],
                     'lr_20170813_1830'  : [85,93,105,109,25,108,81,77,29,37],
                     'gbc_20170813_1800' : [85,93,105,109,25,108,81,77,29,37],
                     # 'gbc_20170813_1830' : [i for i in range(130)],
                     'mlp_20170813_1800' : [85,93,105,109,25,108,81,77,29,37],
                     'ada_20170813_1800' : [85, 93, 105, 109, 25, 108, 81, 77, 29, 37],
                     # 'ada_20170813_1815' : [i for i in range(130)],
                     'xgb_20170813_1800' : [85,93,105,109,25,108,81,77,29,37],
                     # 'xgb_20170813_1830' : [i for i in range(130)],
                     'et_20170813_1800'  : [85,93,105,109,25,108,81,77,29,37],
                     # 'et_20170813_1830'  : [i for i in range(130)]
                    }

print(str(datetime.now()) + ' Reading Data')
with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_unseen, y_unseen = pickle.load(f)
print(str(datetime.now()) + ' Reading Data Complete')

X = []
y = np.array(y_unseen)
# form the training set using the predictions
models_list = sorted([x for x in model_column_dict])
print (models_list)
for model_name in models_list:
    with open('../analysis_graphs/' + str(model_name), 'rb') as f:
        model_list = pickle.load(f)

    X_temp = []
    for i in range(len(model_list)):
        X_temp.append(model_list[i].predict(X_unseen[:,model_column_dict[model_name]]).reshape(-1, 1))

    X_temp = mode(np.hstack((X_temp)), axis = 1)[0].reshape(-1, 1)
    print (X_temp.shape)
    X.append(X_temp)
del X_temp

X = np.hstack((X))
print (X.shape)

y_pred = np.clip(np.sum(X, axis = 1).astype(np.int64), 0, 1)
print (str(skmetrics.accuracy_score(y, y_pred)), str(skmetrics.confusion_matrix(y, y_pred).tolist()), str(skmetrics.roc_auc_score(y, y_pred)))

y_pred = mode(X, axis = 1)[0]
print (str(skmetrics.accuracy_score(y, y_pred)), str(skmetrics.confusion_matrix(y, y_pred).tolist()), str(skmetrics.roc_auc_score(y, y_pred)))


for i in range(len(model_column_dict) - 1):
    for j in range(i + 1, len(model_column_dict)):
        print (str(i), str(j), pearsonr(X[:,i], X[:,j]))

for i in range(len(model_column_dict)):
    print (str(i), pearsonr(X[:,i], y))

'''
train_columns = [i for i in range(X.shape[1])]

param_dict = {'max_depth' : [5, 18, 15],
              'n_estimators' : [120, 300, 500]
              #'min_samples_split' : [2, 5, 10],
              #'min_samples_leaf' : [1, 2, 5, 10]
              }

model_list = []

param_space, param_to_int_dict = c_vars.get_param_space(param_dict)
# param_space = [[5, 2, 5, 120]]
param_space = [[3, 5]]
for param_list in param_space:
    print(str(datetime.now()) + ' Training Random Forest classifier, ' + str(param_list))
    kf = KFold(n_splits = 4, shuffle = True)
    kf_index = 0
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        X_train, X_test = X_train[:, train_columns], X_test[:, train_columns]
        y_train, y_test = y[train_indices], y[test_indices]

        # print (len(X_train), len(X_test), np.sum(y_train), np.sum(y_test))
       
        clf = RandomForestClassifier(max_depth=param_list[param_to_int_dict['max_depth']], 
                                     n_estimators=param_list[param_to_int_dict['n_estimators']],
                                     random_state = 42, class_weight = {0:0.0363, 1:1})

        clf.fit(X_train, y_train)
        model_list.append(clf)
        print (str(datetime.now()) + ' Model Training Complete')

        for myX, myY, Set in zip([X_train, X_test, X_unseen[:, train_columns]], [y_train, y_test, y_unseen], ['Train', 'Test', 'Unseen']):
            y_pred = clf.predict(myX)
            y_pred_proba = clf.predict_proba(myX)

            print (str(datetime.now()) + ' KF_Index,' + Set + ',Accuracy,Confusion_Matrix,AUC')
            print (','.join([str(kf_index), Set, str(skmetrics.accuracy_score(myY, y_pred)), 
                        str(skmetrics.confusion_matrix(myY, y_pred).tolist()), str(skmetrics.roc_auc_score(myY, y_pred_proba[:,1]))]))
                 

        print (clf.feature_importances_)
        kf_index += 1
'''