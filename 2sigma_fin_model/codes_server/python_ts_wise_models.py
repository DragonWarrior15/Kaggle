import numpy as np
import pandas as pd
# from sklearn import linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy.special import cbrt

print ('reading data, will do id test')
X_train = pd.read_csv('data_train.csv')
X_test = pd.read_csv('data_test.csv')

y_train = X_train[['id', 'y']]
y_test = X_test[['id', 'y']]
X_train = X_train[['fundamental_11', 'technical_20', 'technical_30', 'id']]
X_test = X_test[['fundamental_11', 'technical_20', 'technical_30', 'id']]

X_train_mean = X_train.mean(axis = 0)
X_train.fillna(X_train_mean, inplace = True)
X_test.fillna(X_train_mean, inplace = True)

# X_train['fundamental_55_sin'] = X_train['fundamental_55'].apply(lambda x: np.sin(x))
# X_test['fundamental_55_sin'] = X_test['fundamental_55'].apply(lambda x: np.sin(x))

cols_to_use = ['fundamental_11', 'technical_20', 'technical_30']

output_file = open('output_id_wise.txt', 'a')

# data = pd.read_csv('train_small.csv')
# X = data[['fundamental_55', 'technical_20', 'technical_30', 'timestamp']]
# y = data[['y', 'timestamp']]

# print('Splitting into test and train')
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

id_train_list = np.unique(X_train['id'])
model_dict = {}

model_file = open('models.txt', 'w')

for id_key in id_train_list:
  model = Ridge(alpha = 0.07)
  model.fit(X_train[X_train['id'] == id_key][cols_to_use], y_train[y_train['id'] == id_key]['y'])
  
  model_dict[id_key] = model

  model_file.write('id' + str(id_key) + '\n')
  model_file.write('coef_:\n')
  model_file.write(str(model.coef_) + '\n')
  model_file.write('intercept\n')
  model_file.write(str(model.intercept_) + '\n')
  model_file.write('\n')

model_file.close()

print ('calculating score on entire test set')
output_file.write('calculating score on entire test set\n')
y_predicted = []
for i in range(len(y_test)):
  if y_test.iloc[i]['id'] in model_dict:
    y_predicted.append(model_dict[y_test.iloc[i]['id']].predict(X_test.iloc[i][cols_to_use]))
  else:
    # do a nearest neighbor search
    min_dist_index = np.argmin(np.sqrt(np.sum(np.power(np.matrix(X_train[cols_to_use])[:] - np.matrix(X_test.iloc[i][cols_to_use]), 2))))
    y_predicted.append(model_dict[y_train.iloc[min_dist_index]['id']].predict(X_test.iloc[i][cols_to_use]))

y_predicted = np.array(y_predicted)
print ('score : ', np.corrcoef(y_test['y'], y_predicted))
output_file.write(str('score : ' + str(np.corrcoef(y_test['y'], y_predicted)) + '\n'))
'''
print ('calculating score id wise')
output_file.write('calculating score id wise\n')
id_list = np.unique(X_test['timestamp'])
for id_key, index in enumerate(id_list):
  # print ('current id', id_key, 'ids to go', len(id_list) - index - 1)
  y_id = y_test[y_test['timestamp'] == id_key]['y']
  y_predicted_id = y_predicted[y_predicted['timestamp'] == id_key]['y']
  # print ('score', id_key, np.correlate(y_id, predicted_y_id))
  output_file.write(str('score ' + str(id_key) + ' ' + str(np.correlate(y_id, predicted_y_id)) + '\n'))
'''
output_file.close()
print ('done!')
