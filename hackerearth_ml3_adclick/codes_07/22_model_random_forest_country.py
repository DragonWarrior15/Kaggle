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

from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

with open('../analysis_graphs/label_encoder', 'rb') as f:
    label_encoder = pickle.load(f)

with open('../analysis_graphs/standard_scaler', 'rb') as f:
    standard_scaler = pickle.load(f)

print(str(datetime.now()) + ' Reading Data')
with open(c_vars.train_spilt_train_processed, 'rb') as f:
    X, y = pickle.load(f)

country_codes = np.array(standard_scaler.inverse_transform(X)[:,0]).reshape(-1,1).astype(np.int64)

# country_unique = np.unique(X)
# country_unique = np.array(country_unique).reshape(-1,1).astype(np.int64)
# print (country_unique)
# print (label_encoder[0].inverse_transform(country_unique))

n_models = 2
indices_array = [0 for _ in range(n_models)]
indices_array_unseen = [0 for _ in range(n_models)]

indices_array[0] = np.array(list(country_codes == 1)).reshape(-1)
indices_array[1] = np.array(list(country_codes != 1)).reshape(-1)

print (type(indices_array), type(indices_array[0]))
print (X.shape, y.shape, indices_array[0].shape, indices_array[1].shape)

print (indices_array[0].shape, indices_array[1].shape)

X = [X[idx] for idx in indices_array]
y = [y[idx] for idx in indices_array]

print (X[0].shape, X[1].shape)
print (y[0].shape, y[1].shape)


# X_unseen = [X1[:,:] for X1 in X]
# y_unseen = [y1[:,] for y1 in y]


with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_unseen, y_unseen = pickle.load(f)

country_codes = np.array(standard_scaler.inverse_transform(X_unseen)[:,0]).reshape(-1,1).astype(np.int64)

indices_array_unseen[0] = np.array(list(country_codes == 1)).reshape(-1)
indices_array_unseen[1] = np.array(list(country_codes != 1)).reshape(-1)

X_unseen = [X_unseen[idx,:] for idx in indices_array_unseen]
y_unseen = [y_unseen[idx,] for idx in indices_array_unseen]

print(str(datetime.now()) + ' Reading Data Complete')

# param_dict = {'learning_rate' : [0.05, 1, 3],
#               'max_depth' : [5, 10, 50, 100, 200],
#               'n_estimators' : [50, 100, 200]}

param_dict = {'max_depth' : [5, 18, 15],
              'n_estimators' : [120, 300, 500],
              'min_samples_split' : [2, 5, 10],
              'min_samples_leaf' : [1, 2, 5, 10]}

model_list = []

param_space, param_to_int_dict = c_vars.get_param_space(param_dict)
# print (param_space)
# print (param_to_int_dict)
param_space = [[4, 10, 10, 10]]
# param_list = [0.05, 5, 200]
for param_list in param_space:
    # train_columns = c_vars.col_index_training[:c_vars.num_features_for_model]
    train_columns = [x for x in range(X[0].shape[1])]
    print(str(datetime.now()) + ' Training Random Forest classifier, ' + str(param_list))
    kf = [KFold(n_splits = 4, shuffle = True) for _ in range(n_models)]
    clf = [0 for _ in range(n_models)]
    clf[0] = RandomForestClassifier(max_depth=param_list[param_to_int_dict['max_depth']], 
                                    n_estimators=param_list[param_to_int_dict['n_estimators']],
                                    min_samples_split=param_list[param_to_int_dict['min_samples_split']],
                                    min_samples_leaf=param_list[param_to_int_dict['min_samples_leaf']],
                                    random_state = 42)
    clf[1] = RandomForestClassifier(max_depth=param_list[param_to_int_dict['max_depth']], 
                                    n_estimators=param_list[param_to_int_dict['n_estimators']],
                                    min_samples_split=param_list[param_to_int_dict['min_samples_split']],
                                    min_samples_leaf=param_list[param_to_int_dict['min_samples_leaf']],
                                    random_state = 42)   
    kf_index = 0
    print (type(kf[0].split(X[0])))
    for [train_indices_1, test_indices_1], [train_indices_2, test_indices_2] in zip(kf[0].split(X[0]), kf[1].split(X[1])):
        X_train_1, X_test_1 = X[0][train_indices_1], X[0][test_indices_1]
        X_train_2, X_test_2 = X[1][train_indices_2], X[1][test_indices_2]
        X_train_1, X_test_1 = X_train_1[:, train_columns], X_test_1[:, train_columns]
        X_train_2, X_test_2 = X_train_2[:, train_columns], X_test_2[:, train_columns]
        y_train_1, y_test_1 = y[0][train_indices_1], y[0][test_indices_1]
        y_train_2, y_test_2 = y[1][train_indices_2], y[1][test_indices_2]

        clf[0].fit(X_train_1, y_train_1)
        clf[1].fit(X_train_2, y_train_2)
        model_list.append(clf[0])
        model_list.append(clf[1])
        print (str(datetime.now()) + ' Model Training Complete')

        for myX_1, myX_2, myY_1, myY_2, Set_1, Set_2 in zip([X_train_1, X_test_1, X_unseen[0]], [X_train_2, X_test_2, X_unseen[1]],
                                 [y_train_1, y_test_1, y_unseen[0]], [y_train_2, y_test_2, y_unseen[1]],
                                 ['Train1', 'Test1', 'Unseen1'], ['Train2', 'Test2', 'Unseen2']):
            y_pred_1 = clf[0].predict(myX_1)
            y_pred_2 = clf[1].predict(myX_2)
            y_pred_proba_1 = clf[0].predict_proba(myX_1)
            y_pred_proba_2 = clf[1].predict_proba(myX_2)

            print (str(datetime.now()) + ' KF_Index,' + Set_1 + ',' + Set_2 + ',Accuracy1,Accuracy2,CM1,CM2,AUC1,AUC2,Accuracy12,CM12,AUC12')
            print (','.join([str(kf_index), Set_1, Set_2, 
                             str(skmetrics.accuracy_score(myY_1, y_pred_1)), str(skmetrics.accuracy_score(myY_2, y_pred_2)),
                             str(skmetrics.confusion_matrix(myY_1, y_pred_1).tolist()), str(skmetrics.confusion_matrix(myY_2, y_pred_2).tolist()),
                             str(skmetrics.roc_auc_score(myY_1, y_pred_proba_1[:,1])), str(skmetrics.roc_auc_score(myY_2, y_pred_proba_2[:,1])),
                             str(skmetrics.accuracy_score(np.hstack((myY_1,myY_2)), np.hstack((y_pred_1,y_pred_2)))),
                             str(skmetrics.confusion_matrix(np.hstack((myY_1,myY_2)), np.hstack((y_pred_1,y_pred_2))).tolist()),
                             str(skmetrics.roc_auc_score(np.hstack((myY_1,myY_2)), np.hstack((y_pred_proba_1[:,1],y_pred_proba_2[:,1]))))
                             ]))

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