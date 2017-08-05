import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import sklearn.metrics as skmetrics

# np.random.seed(42)

print(str(datetime.now()) + ' Reading Data')
with open(c_vars.train_spilt_train_processed, 'rb') as f:
    X, y = pickle.load(f)

print (X.shape, y.shape)

with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_unseen, y_unseen = pickle.load(f)

print(str(datetime.now()) + ' Reading Data Complete')

param_dict = {'max_depth' : [5, 18, 15],
              'n_estimators' : [120, 300, 500],
              'min_samples_split' : [2, 5, 10],
              'min_samples_leaf' : [1, 2, 5, 10]}

model_list = []

param_space, param_to_int_dict = c_vars.get_param_space(param_dict)
param_space = [[5, 2, 5, 120]]
for param_list in param_space:
    # train_columns = c_vars.col_index_training[:c_vars.num_features_for_model]
    train_columns = [x for x in range(X.shape[1])]
    print(str(datetime.now()) + ' Training Random Forest classifier, ' + str(param_list))
    kf = KFold(n_splits = 4, shuffle = True)
    kf_index = 0
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        X_train, X_test = X_train[:, train_columns], X_test[:, train_columns]
        y_train, y_test = y[train_indices], y[test_indices]
       
        clf = RandomForestClassifier(max_depth=param_list[param_to_int_dict['max_depth']], 
                                     n_estimators=param_list[param_to_int_dict['n_estimators']],
                                     min_samples_split=param_list[param_to_int_dict['min_samples_split']],
                                     min_samples_leaf=param_list[param_to_int_dict['min_samples_leaf']],
                                     random_state = 42)

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

with open('../analysis_graphs/rf_1', 'wb') as f:
    pickle.dump(model_list, f)

print (str(datetime.now()) + ' Reading submit set')
df_submit = pd.read_csv(c_vars.test_file, usecols = ['ID'])
with open(c_vars.test_processed, 'rb') as f:
  X_submit = pickle.load(f)
print (str(datetime.now()) + ' Predicting on submit set')
y_submit_pred = np.sum([0.25 * model_list[i].predict(X_submit) for i in range(4)])
y_submit_pred_proba = clf.predict_proba(X_submit[:,train_columns])
y_submit_pred_proba_1 = y_submit_pred_proba[:,1]
y_submit_pred_proba_1 = np.array([y_submit_pred_proba[i][1] for i in range(len(X_submit))])
df_submit['click'] = y_submit_pred_proba_1

df_submit[['ID', 'click']].to_csv('../output/submit_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv', index = False)
print (str(datetime.now()) + ' Done')
