import pandas as pd
import numpy as np

train_data = '../inputData/train_10to17_train_processed.csv'
test_data = '../inputData/train_18to19_test_processed.csv'
val_data = '../inputData/train_20_val_processed.csv'
submit_data = '../inputData/test_processed.csv'

header_list = ['category', 'merchant', 'offerid', 'siteid',
           'browserid_chrome', 'browserid_ie', 'browserid_firefox', 'browserid_opera', 'browserid_safari', 'browserid_blank',
           'countrycode_a', 'countrycode_b', 'countrycode_c', 'countrycode_d', 'countrycode_e', 'countrycode_f',
           'day_sun', 'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri', 'day_sat',
           'devid_desktop', 'devid_mobile', 'devid_tablet', 'devid_blank',
           'ID', 'click']

# note that click is not there in the final dataset
df_train = pd.read_csv(val_data)
# print(df_train.columns.values)
X_train = df_train[header_list[:-2]].as_matrix()
y_train = df_train[header_list[-1]].as_matrix()

'''
#train a logistic regression model
print('Training a Logistic Regressor')
from sklearn import linear_model
h = .02  # step size in the mesh
clf = linear_model.LogisticRegression(C=1e5, max_iter = 1)
'''

#train a adaboost classifier
print('Training a AdaBoost Classifier')
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=20)

clf.fit(X_train, y_train)
print ('Model Training Complete')

df_test = pd.read_csv(test_data)

X_test = df_test[header_list[:-2]].as_matrix()
y_test = df_test[header_list[-1]].as_matrix()

# predict on the test set
y_test_pred = clf.predict(X_test)
y_test_pred_proba = clf.predict_proba(X_test)
# y_test_pred_proba_1 = np.array([y_test_pred_proba[i][1] for i in range(len(y_test))])
y_test_pred_proba_1 = np.array(y_test_pred_proba[:][1])

# get the accuracies and relevant metrics
print('Getting Scores on test set')
import sklearn.metrics as skmetrics
print('Accuracy Score : ' + str(skmetrics.accuracy_score(y_test, y_test_pred)))
print('Confusion Matrix : ' + str(skmetrics.confusion_matrix(y_test, y_test_pred)))
print('AUC score : ' + str(skmetrics.roc_auc_score(y_test, y_test_pred_proba_1)))