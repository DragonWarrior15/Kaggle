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

from xgboost.sklearn import XGBClassifier

np.random.seed(42)

print(str(datetime.now()) + ' Reading Data')
with open(c_vars.train_spilt_train_processed, 'rb') as f:
    X, y = pickle.load(f)

with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_unseen, y_unseen = pickle.load(f)

print(str(datetime.now()) + ' Reading Data Complete')

# param_dict = {'learning_rate' : [0.05, 1, 3],
#               'max_depth' : [5, 10, 50, 100, 200],
#               'n_estimators' : [50, 100, 200]}

param_dict = {'max_depth' : [5, 18, 15],
              'min_child_weight' : [1, 3, 5, 7],
              'subsample' : [0.6, 0.8, 1],
              'colsample_bytree' : [0.6, 0.8, 1]
              }

model_list = []

param_space, param_to_int_dict = c_vars.get_param_space(param_dict)
# print (param_space)
# print (param_to_int_dict)
param_space = [[0.6, 5, 1, 0.6]]
# param_list = [0.05, 5, 200]
for param_list in param_space:
    # train_columns = c_vars.col_index_training[:c_vars.num_features_for_model]
    train_columns = [x for x in range(X.shape[1])]
    print(str(datetime.now()) + ' Training Random Forest classifier, ' + str(param_list))
    kf = KFold(n_splits = 4, shuffle = True)
    clf = XGBClassifier(max_depth        = param_list[param_to_int_dict['max_depth']], 
                        min_child_weight = param_list[param_to_int_dict['min_child_weight']],
                        subsample        = param_list[param_to_int_dict['subsample']],
                        colsample_bytree = param_list[param_to_int_dict['colsample_bytree']]
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

'''
df_submit = pd.read_csv(c_vars.test_file)
with open(c_vars.test_processed, 'rb') as f:
  X_submit = pickle.load(f)
print (str(datetime.now()) + ' Predicting on submit set')
y_submit_pred = np.sum([0.25 * model_list[i].predict(X_submit) for i in range(4)])
y_submit_pred_proba = clf.predict_proba(X_submit[:,c_vars.col_index_training[:c_vars.num_features_for_model]])
y_submit_pred_proba_1 = y_submit_pred_proba[:,1]
y_submit_pred_proba_1 = np.array([y_submit_pred_proba[i][1] for i in range(len(X_submit))])
df_submit['click'] = y_submit_pred_proba_1

df_submit[['ID', 'click']].to_csv('../output/submit_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv', index = False)
print (str(datetime.now()) + ' Done')
'''