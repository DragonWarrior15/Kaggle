import numpy as np
import pandas as pd
# from sklearn import linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from scipy.special import cbrt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans

print ('reading data, will do id test')
X_train = pd.read_csv('data_train.csv')
# X_test = pd.read_csv('data_test.csv')

y_train = X_train[['id', 'y']]
# y_test = X_test[['id', 'y']]
# X_train = X_train[['fundamental_11', 'technical_20', 'technical_30', 'id']]
# X_test = X_test[['fundamental_11', 'technical_20', 'technical_30', 'id']]

output_file = open('output_id_wise.txt', 'a')

# data = pd.read_csv('train_small.csv')
# X = data[['fundamental_55', 'technical_20', 'technical_30', 'timestamp']]
# y = data[['y', 'timestamp']]

# print('Splitting into test and train')
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
default_mean = X_train.mean(axis = 0)

# X_test.fillna(default_mean, inplace = True)
X_train.fillna(default_mean, inplace = True)

model_file = open('models.txt', 'w')

model_dict = {}

cols_to_use = list(X_train.columns.values)
# print (cols_to_use)
cols_to_use.remove('id')
cols_to_use.remove('timestamp')
cols_to_use.remove('y')

model_cluster = KMeans(n_clusters = 10, init = 'k-means++', n_init=10, random_state = 42, n_jobs = -1)
model_cluster.fit(X_train[cols_to_use])

X_train['cluster'] = model_cluster.labels_ 

print (X_train['cluster'].head(5))

cluster_id = np.unique(model_cluster.labels_)

for cluster_index in cluster_id:
  X_train[X_train['cluster'] == cluster_index].corr().to_csv('cluster_corr/cluster_' + str(cluster_index) + '.csv')

'''

model_file.close()

default_model = Ridge(alpha = 0.07)
default_model.fit(X_train[['technical_20', 'technical_30', 'fundamental_11']], y_train['y'])

print ('calculating score on entire test set')
output_file.write('calculating score on entire test set\n')
y_predicted = []
for i in range(len(y_test)):
  if y_test.iloc[i]['id'] in model_dict:
    y_predicted.append(model_dict[y_test.iloc[i]['id']].predict(X_test.iloc[i][cols_to_use]))
  else:
    # do a nearest neighbor search
    # min_dist_index = np.argmin(np.sqrt(np.sum(np.power(np.matrix(X_train[id_col_dict[id_key]])[:] - np.matrix(X_test.iloc[i][id_col_dict[id_key]]), 2))))
    # y_predicted.append(model_dict[y_train.iloc[min_dist_index]['id']].predict(X_test.iloc[i][id_col_dict[id_key]]))
    print (X_test.iloc[i][['technical_20', 'technical_30', 'fundamental_11']])
    y_predicted.append(default_model.predict(X_test.iloc[i][['technical_20', 'technical_30', 'fundamental_11']]))

y_predicted = np.array(y_predicted)
print ('score : ', np.corrcoef(y_test['y'], y_predicted))
output_file.write(str('score : ' + str(np.corrcoef(y_test['y'], y_predicted)) + '\n'))

print ('calculating score id wise')
output_file.write('calculating score id wise\n')
id_list = np.unique(X_test['timestamp'])
for id_key, index in enumerate(id_list):
  # print ('current id', id_key, 'ids to go', len(id_list) - index - 1)
  y_id = y_test[y_test['timestamp'] == id_key]['y']
  y_predicted_id = y_predicted[y_predicted['timestamp'] == id_key]['y']
  # print ('score', id_key, np.correlate(y_id, predicted_y_id))
  output_file.write(str('score ' + str(id_key) + ' ' + str(np.correlate(y_id, predicted_y_id)) + '\n'))

output_file.close()
print ('done!')
'''