import numpy as np
import pandas as pd
# from sklearn import linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy.special import cbrt

print ('reading data, will do id test')
X_train = pd.read_csv('train_small.csv')
X_test = pd.read_csv('data_test.csv')

y_train = X_train[['id', 'y']]
y_test = X_test[['id', 'y']]
X_train = X_train[['fundamental_55', 'technical_20', 'technical_30', 'id']]
X_test = X_test[['fundamental_55', 'technical_20', 'technical_30', 'id']]

X_train_mean = X_train.mean(axis = 0)
X_train.fillna(X_train_mean, inplace = True)
X_test.fillna(X_train_mean, inplace = True)

X_train['fundamental_55_sin'] = X_train['fundamental_55'].apply(lambda x: np.sin(x))
X_test['fundamental_55_sin'] = X_test['fundamental_55'].apply(lambda x: np.sin(x))

cols_to_use = ['fundamental_55_sin', 'technical_20', 'technical_30']

for i in [0, 0.1, 0.5, 1, 10, 100]:
  output_file = open('output_id_wise.txt', 'w')

  # data = pd.read_csv('train_small.csv')
  # X = data[['fundamental_55', 'technical_20', 'technical_30', 'id']]
  # y = data[['y', 'id']]

  # print('Splitting into test and train')
  # from sklearn.cross_validation import train_test_split
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  print ('training model, alpha is : ' + str(i))
  model = Ridge(alpha = i)
  model.fit(X_train[cols_to_use], y_train['y'])

  print ('calculating score on entire test set')
  output_file.write('calculating score on entire test set\n')
  y_predicted = model.predict(X_test[cols_to_use])

  print ('score : ', np.correlate(y_test['y'], y_predicted))
  output_file.write(str('score : ' + str(np.correlate(y_test['y'], y_predicted)) + '\n'))

  print ('calculating score id wise')
  output_file.write('calculating score id wise\n')
  id_list = np.unique(X_test['id'])
  for id_key, index in enumerate(id_list):
    # print ('current id', id_key, 'ids to go', len(id_list) - index - 1)
    y_id = y_test[y_test['id'] == id_key]['y']
    predicted_y_id = model.predict(X_test[X_test['id'] == id_key][cols_to_use])
    # print ('score', id_key, np.correlate(y_id, predicted_y_id))
    output_file.write(str('score ' + str(id_key) + ' ' + str(np.correlate(y_id, predicted_y_id)) + '\n'))

  output_file.close()
  print ('done!')