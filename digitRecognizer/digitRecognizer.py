# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# read the files and get X, y
print('Reading train.csv')
import csv
f = open('train.csv', 'r')
csv_reader_obj = csv.reader(f)
train_data = []
for row in csv_reader_obj:
    train_data.append(row)
f.close()
X = [train_data[i][1:] for i in range(1, len(train_data))]
y = [train_data[i][0] for i in range(1, len(train_data))]

'''
# to check length of the X and y arrays
print(len(X))
print(len(X[0]))
print(len(train_data[0]))
print(len(y))
'''
'''
# fit a pca to reduce the dimension, PCA doesnt seem to be improving the performance
print('Reducing dimensions with PCA')
from sklearn.decomposition import PCA
X = PCA(n_components = 112).fit_transform(X)
'''

# split X and y into training and tst data set for training and evaluating the model
print('Splitting into test and train')
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train a random forest classifier
print('Training Random Forest Classifier')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 30, max_depth = 30)
clf.fit(X_train, y_train)

# predict on X_test
y_test_pred = clf.predict(X_test)

# get the accuracies and relevant metrics
print('Getting Scores on test set')
import sklearn.metrics as skmetrics
print('Accuracy Score : ' + str(skmetrics.accuracy_score(y_test, y_test_pred)))
#print('Confusion Matrix : ' + str(metrics.confusion_matrix(y_test, y_test_pred)))

# import the test dataset on which we have to output the results of our model
print('reading test.csv')
f = open('test.csv', 'r')
csv_reader_obj = csv.reader(f)
train_data = []
for row in csv_reader_obj:
    train_data.append(row)
X_test_dataset = [train_data[i][:] for i in range(1,len(train_data))]

# use when reducing the features with PCA 
# X_test_dataset = PCA(n_components = 112).fit_transform(X_test_dataset)

# predict the classification for test dataset
print('Predicting on test dataset')
y_test_dataset = clf.predict(X_test_dataset)

# write outputs to file
print('Writing results to file')
f = open('digitRecognizerOutput.csv', 'w')
f.write('ImageId,Label\n')
for index, item in enumerate(y_test_dataset):
    f.write(str(index + 1) + ',' + str(item) + '\n')
f.close()

print('Done!')