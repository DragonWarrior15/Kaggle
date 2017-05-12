import pandas as pd
import random
import numpy as np


keeplist = ['outcome', 'pattr_char_38', 'woe_pattr_char_2', 'woe_pattr_group_1']

pd.concat([pd.read_csv('merged_dataset_train_0.1_type1_mod.csv')[keeplist], pd.read_csv('merged_dataset_train_0.1_not_type1_mod.csv')[keeplist]]).to_csv('merged_dataset_train_mod_part.csv', index = False)

print('Reading train set.csv')
import csv
f = open('merged_dataset_train_mod_part.csv', 'r')
csv_reader_obj = csv.reader(f)
train_data = []
for row in csv_reader_obj:
    train_data.append(row)
f.close()
X = [train_data[i][1:] for i in range(1, len(train_data))]
y = [train_data[i][0] for i in range(1, len(train_data))]

print('Splitting into test and train')
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = np.array(X_train).astype(float, casting = 'unsafe')
X_test = np.array(X_test).astype(float, casting = 'unsafe')

from sklearn.preprocessing import normalize
normalize(X_train, copy = False)
normalize(X_test, copy = False)

y_train = np.array(y_train).astype(int, casting = 'unsafe')
y_test = np.array(y_test).astype(int, casting = 'unsafe')

'''
#train a neural network
print('Training a neural net')
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(25, 10), random_state=1)
'''

'''
# train a random forest classifier
print('Training Random Forest Classifier')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 20, max_depth = 20)
'''

'''
# train a svm classifier
print('Training SVM')
from sklearn import svm
clf = svm.SVC()
'''


#train a logistic regression model
print('Training a Logistic Regressor')
from sklearn import linear_model
h = .02  # step size in the mesh
clf = linear_model.LogisticRegression(C=1e5)


'''
#train a adaboost classifier
print('Training a AdaBoost Classifier')
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=40), algorithm="SAMME", n_estimators=200)
'''

'''
# train xgboost model
import xgboost as xgb
print('Training xgboost classifier')
clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
'''


# predict on X_test
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
y_test_pred_proba = clf.predict_proba(X_test)
y_test_pred_proba_1 = np.array([y_test_pred_proba[i][1] for i in range(len(y_test))])


# get the accuracies and relevant metrics
print('Getting Scores on test set')
import sklearn.metrics as skmetrics
print('Accuracy Score : ' + str(skmetrics.accuracy_score(y_test, y_test_pred)))
print('Confusion Matrix : ' + str(skmetrics.confusion_matrix(y_test, y_test_pred)))
print('AUC score : ' + str(skmetrics.roc_auc_score(y_test, y_test_pred_proba_1)))




keeplist.remove('outcome')
pd.read_csv('merged_dataset_test_type1_mod.csv')[keeplist].to_csv('merged_dataset_test_type1_mod_part.csv', index = False)


print('Reading test set.csv')
import csv
f = open('merged_dataset_test_type1_mod_part.csv', 'r')
csv_reader_obj = csv.reader(f)
test_data = []
for row in csv_reader_obj:
    test_data.append(row)
f.close()

X_test = [test_data[i][:] for i in range(1, len(test_data))]
X_test = np.array(X_test).astype(float, casting = 'unsafe')
normalize(X_test, copy = False)
y_pred = clf.predict(X_test)

f = open('submit_lr_type1_20160904_1958.csv', 'w')
f.write('activity_id,outcome\n')
for i in range(len(y_pred)):
	f.write(str(test_data[i+1][0]) + ',' + str(y_pred[i]) + '\n')
f.close()
