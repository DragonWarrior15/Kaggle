import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
import pickle
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

print(str(datetime.now()) + ' Reading Data')
with open(c_vars.train_spilt_train_processed, 'rb') as f:
    X, y = pickle.load(f)
with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_test, y_test = pickle.load(f)
print(str(datetime.now()) + ' Reading Data Complete')

param_dict = {'max_depth' : [5, 18, 15],
              'n_estimators' : [120, 300, 500],
              'min_samples_split' : [2, 5, 10],
              'min_samples_leaf' : [1, 2, 5, 10]}

param_space, param_to_int_dict = c_vars.get_param_space(param_dict)
param_space = [[5, 2, 5, 120]]
for param_list in param_space:
    # train xgboost model
    # import xgboost as xgb
    print(str(datetime.now()) + ' Training Random Forest classifier, ' + str(param_list))
    # print (param_list[param_to_int_dict['learning_rate']], param_list[param_to_int_dict['max_depth']], param_list[param_to_int_dict['n_estimators']])
    clf = RandomForestClassifier(max_depth=param_list[param_to_int_dict['max_depth']], 
                                 n_estimators=param_list[param_to_int_dict['n_estimators']],
                                 min_samples_split=param_list[param_to_int_dict['min_samples_split']],
                                 min_samples_leaf=param_list[param_to_int_dict['min_samples_leaf']])

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
# y_submit_pred = clf.predict(X_submit)
y_submit_pred_proba = clf.predict_proba(X_submit)
y_submit_pred_proba_1 = y_submit_pred_proba[:,1]
# y_submit_pred_proba_1 = np.array([y_submit_pred_proba[i][1] for i in range(len(X_submit))])
df_submit['click'] = y_submit_pred_proba_1

df_submit[['ID', 'click']].to_csv('../output/submit_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv', index = False)
print (str(datetime.now()) + ' Done')
