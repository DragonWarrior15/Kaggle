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

# df = pd.read_csv(c_vars.train_sample_file)
df = pd.read_csv(c_vars.train_split_train_sample)
# df = pd.read_csv(c_vars.train_split_train)
df = df[c_vars.header_useful]

df.fillna(c_vars.fillna_dict, inplace = True)

df.loc[:, 'datetime'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
df.loc[:, 'browserid'] = df['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           else c_vars.browserid_map[x])

X = df[c_vars.header_useful[:-1]].as_matrix()
y = df['click'].as_matrix()

with open('../analysis_graphs/label_encoder', 'rb') as f:
    label_encoder = pickle.load(f)
with open('../analysis_graphs/one_hot_encoding', 'rb') as f:
    ohe = pickle.load(f)

print (str(datetime.now()) + ' Label Encoding Started')
for i in range(len(label_encoder)):
    X[:,i] = label_encoder[i].transform(X[:,i])
print (str(datetime.now()) + ' Label Encoding Completed')

print (str(datetime.now()) + ' One Hot Encoding Started')
X_ohe = ohe.transform(X[:,[0,5,6,7]])
print (str(datetime.now()) + ' One Hot Encoding Complete')

X = np.hstack((X[:,[i for i in range(len(c_vars.header_useful)-1) if i not in [0,5,6,7]]], X_ohe))

X = X[:,c_vars.col_index_training]

print (X.shape, y.shape)
sm = SMOTE(random_state=42)
X, y = sm.fit_sample(X, y)

df_test = pd.read_csv(c_vars.train_split_val)
df_test.fillna(c_vars.fillna_dict, inplace = True)
df_test.loc[:, 'datetime'] = df_test['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
df_test.loc[:, 'browserid'] = df_test['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           else c_vars.browserid_map[x])
X_test = df_test[c_vars.header_useful[:-1]].as_matrix()
y_test = df_test['click'].as_matrix()
for i in range(len(label_encoder)):
    X_test[:,i] = label_encoder[i].transform(X_test[:,i])
X_test_ohe = ohe.transform(X_test[:,[0,5,6,7]])
X_test = np.hstack((X_test[:,[i for i in range(len(c_vars.header_useful)-1) if i not in [0,5,6,7]]], X_test_ohe))
X_test = X_test[:,c_vars.col_index_training]

param_dict = {'learning_rate' : [0.05, 1, 3],
              'max_depth' : [5, 10, 50, 100, 200],
              'n_estimators' : [50, 100, 200]}

param_space = []
param_list = sorted(list([k for k in param_dict]))
param_to_int_dict = dict([[param_list[i], i] for i in range(len(param_list))])
# print (param_to_int_dict)
for param in param_list:
    curr_param_space_length = len(param_space)
    if (curr_param_space_length == 0):
        for i in range(len(param_dict[param])):
            param_space.append([param_dict[param][i]])
    else:
        for i in range(len(param_dict[param]) - 1):
            for j in range(curr_param_space_length):
                param_space.append(list(param_space[j]) + [param_dict[param][i]])

        for i in range(curr_param_space_length):
            param_space[i].append(param_dict[param][-1])

# print (param_space)
param_space = sorted(param_space)
param_space = [[0.05, 5, 200]]
for param_list in param_space:
    # train xgboost model
    # import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    print(str(datetime.now()) + ' Training xgboost classifier, ' + str(param_list))
    # print (param_list[param_to_int_dict['learning_rate']], param_list[param_to_int_dict['max_depth']], param_list[param_to_int_dict['n_estimators']])
    clf = XGBClassifier(max_depth=param_list[param_to_int_dict['max_depth']], 
                        n_estimators=param_list[param_to_int_dict['n_estimators']],
                        learning_rate=param_list[param_to_int_dict['learning_rate']])


    clf.fit(X, y)
    print (str(datetime.now()) + ' Model Training Complete')


    print (str(datetime.now()) + ' Making Predictions on test set')
    # predict on the test set
    y_test_pred = clf.predict(X_test)
    y_test_pred_proba = clf.predict_proba(X_test)
    y_test_pred_proba_1 = np.array([y_test_pred_proba[i][1] for i in range(len(y_test))])
    # y_test_pred_proba_1 = np.array(y_test_pred_proba[:][1])

    # get the accuracies and relevant metrics
    print(str(datetime.now()) + ' Getting Scores on test set')
    import sklearn.metrics as skmetrics
    print(str(datetime.now()) + ' Accuracy Score : ' + str(skmetrics.accuracy_score(y_test, y_test_pred)))
    print(str(datetime.now()) + ' Confusion Matrix : ' + str(skmetrics.confusion_matrix(y_test, y_test_pred)))
    print(str(datetime.now()) + ' AUC score : ' + str(skmetrics.roc_auc_score(y_test, y_test_pred_proba_1)))

print (str(datetime.now()) + ' Transforming submit set')
# submit set
df_submit = pd.read_csv(c_vars.test_file)
df_submit.fillna(c_vars.fillna_dict, inplace = True)
df_submit.loc[:, 'datetime'] = df_submit['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
df_submit.loc[:, 'browserid'] = df_submit['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           else c_vars.browserid_map[x])
X_submit = df_submit[c_vars.header_useful[:-1]].as_matrix()
for i in range(len(label_encoder)):
    X_submit[:,i] = label_encoder[i].transform(X_submit[:,i])
X_submit_ohe = ohe.transform(X_submit[:,[0,5,6,7]])
X_submit = np.hstack((X_submit[:,[i for i in range(len(c_vars.header_useful)-1) if i not in [0,5,6,7]]], X_submit_ohe))
X_submit = X_submit[:,c_vars.col_index_training]

print (str(datetime.now()) + ' Predicting on submit set')
# y_submit_pred = clf.predict(X_submit)
y_submit_pred_proba = clf.predict_proba(X_submit)
y_submit_pred_proba_1 = y_submit_pred_proba[:,1]
# y_submit_pred_proba_1 = np.array([y_submit_pred_proba[i][1] for i in range(len(X_submit))])
df_submit['click'] = y_submit_pred_proba_1

df_submit[['ID', 'click']].to_csv('../output/submit_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv', index = False)
print (str(datetime.now()) + ' Done')