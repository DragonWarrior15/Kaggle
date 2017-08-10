import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import sklearn.metrics as skmetrics

# np.random.seed(42)
# train_columns = [i for i in range(69)]
# train_columns = [i for i in range(69)]
# train_columns = [64,56,23,67,48,44,59,28,60,52,62,63,16,27,51]
# train_columns = [64,60,24,28,68,23,27,56,44,22,52,21,48,26,25,47,45,46,12,9,10,11]
# train_columns = [64,60,24,28,68,23,27,56,44,22,52]
train_columns = [64,60,24,28,16,32,54,55,36,42]#,43,0,8,67,59,63,35,4,51,30,6,40,18,15,3]
print (train_columns)
'''
print(str(datetime.now()) + ' Reading Data')
with open(c_vars.train_spilt_train_processed, 'rb') as f:
    X, y = pickle.load(f)

print (X.shape, y.shape)

with open(c_vars.train_spilt_val_processed, 'rb') as f:
    X_unseen, y_unseen = pickle.load(f)

print(str(datetime.now()) + ' Reading Data Complete')

param_dict = {'penalty' : ['l1', 'l2'],
              'C' : [0.1, 1, 10]}

model_list = []

param_space, param_to_int_dict = c_vars.get_param_space(param_dict)
# param_space = [[5, 2, 5, 120]]
param_space = [[10, 'l2']]
for param_list in param_space:
    # train_columns = c_vars.col_index_training[:c_vars.num_features_for_model]
    # train_columns = [x for x in range(X.shape[1])]
    print(str(datetime.now()) + ' Training Logistic Regression classifier, ' + str(param_list))
    kf = KFold(n_splits = 4, shuffle = True)
    kf_index = 0
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        X_train, X_test = X_train[:, train_columns], X_test[:, train_columns]
        y_train, y_test = y[train_indices], y[test_indices]
       
        clf = LogisticRegression(penalty=param_list[param_to_int_dict['penalty']],
                                     C=param_list[param_to_int_dict['C']],
                                     random_state = 42)

        clf.fit(X_train, y_train)
        model_list.append(clf)
        print (str(datetime.now()) + ' Model Training Complete')

        for myX, myY, Set in zip([X_train, X_test, X_unseen[:, train_columns]], [y_train, y_test, y_unseen], ['Train', 'Test', 'Unseen']):
            y_pred = clf.predict(myX)
            y_pred_proba = clf.predict_proba(myX)

            print (str(datetime.now()) + ' KF_Index,' + Set + ',Accuracy,Confusion_Matrix,AUC')
            print (','.join([str(kf_index), Set, str(skmetrics.accuracy_score(myY, y_pred)), 
                        str(skmetrics.confusion_matrix(myY, y_pred).tolist()), str(skmetrics.roc_auc_score(myY, y_pred_proba[:,1]))]))
            
            # myY_Y = myY
            # myX_X = myX
            # if Set == 'Unseen':
                # print ('day wise scores')
                # for lower, higher, i in zip([0.4, 0.9, 1.4], [0.7, 1.2, 1.7], [5,4,6]):
                    # myY = myY_Y[(lower <= myX_X[:,3]) & (myX_X[:,3] <= higher)]
                    # myX = myX_X[(lower <= myX_X[:,3]) & (myX_X[:,3] <= higher), :]
                    # print (len(myY), np.unique(myX[:,3]))
                    # y_pred = clf.predict(myX)
                    # y_pred_proba = clf.predict_proba(myX)
                    # print ('day_ind, ' + str(datetime.now()) + ' KF_Index,' + Set + ',Accuracy,Confusion_Matrix,AUC')
                    # print (','.join([str(i), Set, str(skmetrics.accuracy_score(myY, y_pred)), 
                        # str(skmetrics.confusion_matrix(myY, y_pred).tolist()), str(skmetrics.roc_auc_score(myY, y_pred_proba[:,1]))]))      

        print (clf.coef_)
        kf_index += 1

with open('../analysis_graphs/lr_20170810_0120', 'wb') as f:
    pickle.dump(model_list, f)

# rf_20170809_1212 0.63508 LB
# lr_20170809_1622 0.67096 LB
# lr_20170809_1147 0.66888 LB, [64,60,24,28,16,32,54,55,36,42]
'''
with open('../analysis_graphs/lr_20170810_0120', 'rb') as f:
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
