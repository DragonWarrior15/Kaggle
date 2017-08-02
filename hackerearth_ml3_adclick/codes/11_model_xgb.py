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
from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier

np.random.seed(42)

print(str(datetime.now()) + ' Reading Data')
with open(c_vars.train_spilt_train_processed, 'rb') as f:
    X, y = pickle.load(f)
with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_test, y_test = pickle.load(f)
print(str(datetime.now()) + ' Reading Data Complete')

# param_dict = {'learning_rate' : [0.05, 1, 3],
#               'max_depth' : [5, 10, 50, 100, 200],
#               'n_estimators' : [50, 100, 200]}

param_dict = {'max_depth' : [5, 18, 15],
              'min_child_weight' : [1, 3, 5, 7],
              'subsample' : [0.6, 0.8, 1],
              'colsample_bytree' : [0.6, 0.8, 1]
              }

param_space, param_to_int_dict = c_vars.get_param_space(param_dict)
print (param_space)
print (param_to_int_dict)
param_space = [[0.6, 5, 1, 0.6]]
# param_list = [0.05, 5, 200]
for param_list in param_space:
    # i = len(c_vars.col_index_training)
    # for i in range(1, len(c_vars.col_index_training)):
    # print (c_vars.col_index_training[:i])
    # train xgboost model
    # import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    print(str(datetime.now()) + ' Training xgboost classifier, ' + str(param_list))
    # print (param_list[param_to_int_dict['learning_rate']], param_list[param_to_int_dict['max_depth']], param_list[param_to_int_dict['n_estimators']])
    clf = XGBClassifier(max_depth        = param_list[param_to_int_dict['max_depth']], 
                        min_child_weight = param_list[param_to_int_dict['min_child_weight']],
                        subsample        = param_list[param_to_int_dict['subsample']],
                        colsample_bytree = param_list[param_to_int_dict['colsample_bytree']]
                        )


    clf.fit(X[:,c_vars.col_index_training[:c_vars.num_features_for_model]], y)
    print (str(datetime.now()) + ' Model Training Complete')


    print (str(datetime.now()) + ' Making Predictions on train set')
    # predict on the test set
    y_pred = clf.predict(X[:,c_vars.col_index_training[:c_vars.num_features_for_model]])
    y_pred_proba = clf.predict_proba(X[:,c_vars.col_index_training[:c_vars.num_features_for_model]])
    y_pred_proba_1 = y_pred_proba[:,1]

    # get the accuracies and relevant metrics
    print(str(datetime.now()) + ' Getting Scores on train set')
    import sklearn.metrics as skmetrics
    print(str(datetime.now()) + ' Accuracy Score : ' + str(skmetrics.accuracy_score(y, y_pred)))
    print(str(datetime.now()) + ' Confusion Matrix : ' + str(skmetrics.confusion_matrix(y, y_pred)))
    print(str(datetime.now()) + ' AUC score : ' + str(skmetrics.roc_auc_score(y, y_pred_proba_1)))
    

    print (str(datetime.now()) + ' Making Predictions on test set')
    # predict on the test set
    y_test_pred = clf.predict(X_test[:,c_vars.col_index_training[:c_vars.num_features_for_model]])
    y_test_pred_proba = clf.predict_proba(X_test[:,c_vars.col_index_training[:c_vars.num_features_for_model]])
    y_test_pred_proba_1 = y_test_pred_proba[:,1]

    # get the accuracies and relevant metrics
    print(str(datetime.now()) + ' Getting Scores on test set')
    import sklearn.metrics as skmetrics
    print(str(datetime.now()) + ' Accuracy Score : ' + str(skmetrics.accuracy_score(y_test, y_test_pred)))
    print(str(datetime.now()) + ' Confusion Matrix : ' + str(skmetrics.confusion_matrix(y_test, y_test_pred)))
    print(str(datetime.now()) + ' AUC score : ' + str(skmetrics.roc_auc_score(y_test, y_test_pred_proba_1)))


df_submit = pd.read_csv(c_vars.test_file)
with open(c_vars.test_processed, 'rb') as f:
  X_submit = pickle.load(f)
print (str(datetime.now()) + ' Predicting on submit set')
y_submit_pred = clf.predict(X_submit)
y_submit_pred_proba = clf.predict_proba(X_submit[:,c_vars.col_index_training[:c_vars.num_features_for_model]])
y_submit_pred_proba_1 = y_submit_pred_proba[:,1]
y_submit_pred_proba_1 = np.array([y_submit_pred_proba[i][1] for i in range(len(X_submit))])
df_submit['click'] = y_submit_pred_proba_1

df_submit[['ID', 'click']].to_csv('../output/submit_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv', index = False)
print (str(datetime.now()) + ' Done')