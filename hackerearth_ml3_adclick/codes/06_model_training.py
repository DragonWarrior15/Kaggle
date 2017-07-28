import pandas as pd
import numpy as np
from datetime import datetime

train_data = '../inputData/train_10to18_train_processed.csv'
train_data = '../inputData/train_10to18_train_processed_sample.csv'
test_data = '../inputData/train_19to20_test_processed.csv'
submit_data = '../inputData/test_processed.csv'

header_list = ['category', 'merchant', 'offerid', 'siteid',
           'browserid_chrome', 'browserid_ie', 'browserid_firefox', 'browserid_opera', 'browserid_safari', 'browserid_blank',
           'countrycode_a', 'countrycode_b', 'countrycode_c', 'countrycode_d', 'countrycode_e', 'countrycode_f',
           'day_sun', 'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri', 'day_sat',
           'devid_desktop', 'devid_mobile', 'devid_tablet', 'devid_blank',
           'ID', 'click']

# note that click is not there in the final dataset
df_train = pd.read_csv(train_data)
# print(df_train.columns.values)
X_train = df_train[header_list[:-2]].as_matrix()
y_train = df_train[header_list[-1]].as_matrix()

'''
#train a logistic regression model
print('Training a Logistic Regressor')
from sklearn import linear_model
h = .02  # step size in the mesh
clf = linear_model.LogisticRegression(C=1e5)
'''
'''
#train a adaboost classifier
print('Training a AdaBoost Classifier')
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=20)
'''
'''
# train a random forest classifier
print('Training Random Forest Classifier')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 40, max_depth = 10, random_state = 42)
'''
np.random.seed(42)
param_dict = {'learning_rate' : [0.05, 1],
              'max_depth' : [5, 10],
              'n_estimators' : [50, 100]}
# param_dict = {'learning_rate' : [0.05, 1, 2, 3],
#               'max_depth' : [5, 10, 30, 50],
#               'n_estimators' : [50, 100, 200, 300, 500]}

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
                param_space.append(list(param_space[j]))

        for i in range(len(param_space)):
            param_space[i].append(param_dict[param][i%len(param_dict[param])])

# print (param_space)

for param_list in param_space:
    # train xgboost model
    # import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    print('Training xgboost classifier, ' + str(param_list))
    # print (param_list[param_to_int_dict['learning_rate']], param_list[param_to_int_dict['max_depth']], param_list[param_to_int_dict['n_estimators']])
    clf = XGBClassifier(max_depth=param_list[param_to_int_dict['max_depth']], 
                            n_estimators=param_list[param_to_int_dict['n_estimators']],
                            learning_rate=param_list[param_to_int_dict['learning_rate']])


    clf.fit(X_train, y_train)
    print ('Model Training Complete')

    df_test = pd.read_csv(test_data)

    X_test = df_test[header_list[:-2]].as_matrix()
    y_test = df_test[header_list[-1]].as_matrix()

    print ('Making Predictions on test set')
    # predict on the test set
    y_test_pred = clf.predict(X_test)
    y_test_pred_proba = clf.predict_proba(X_test)
    y_test_pred_proba_1 = np.array([y_test_pred_proba[i][1] for i in range(len(y_test))])
    # y_test_pred_proba_1 = np.array(y_test_pred_proba[:][1])

    # get the accuracies and relevant metrics
    print('Getting Scores on test set')
    import sklearn.metrics as skmetrics
    print('Accuracy Score : ' + str(skmetrics.accuracy_score(y_test, y_test_pred)))
    print('Confusion Matrix : ' + str(skmetrics.confusion_matrix(y_test, y_test_pred)))
    print('AUC score : ' + str(skmetrics.roc_auc_score(y_test, y_test_pred_proba_1)))

'''
print ('Making Predictions on submit set')
df_submit = pd.read_csv(submit_data)

X_submit = df_submit[header_list[:-2]].as_matrix()
# y_test = df_test[header_list[-1]].as_matrix()
y_submit_pred = clf.predict(X_submit)
y_submit_pred_proba = clf.predict_proba(X_submit)
y_submit_pred_proba_1 = np.array([y_submit_pred_proba[i][1] for i in range(len(y_submit_pred))])

df_submit['click'] = y_submit_pred_proba_1

df_submit[['ID', 'click']].to_csv('../output/submit_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv', index = False)
'''