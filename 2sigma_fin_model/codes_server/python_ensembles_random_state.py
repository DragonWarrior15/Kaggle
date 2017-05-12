import numpy as np
import pandas as pd
# from sklearn import linear_model as lm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy.special import cbrt
from sklearn.ensemble import ExtraTreesRegressor

# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

print ('reading data, will do id test')
X_train = pd.read_csv('train_small.csv')
X_test = pd.read_csv('test_small.csv')

y_is_above_cut = (X_train['y'] > high_y_cut)
y_is_below_cut = (X_train['y'] < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

cols_to_use_2 = list(X_train.columns)
cols_to_use_2.remove('id')
cols_to_use_2.remove('y')

cols_to_use = ['fundamental_11', 'technical_20', 'technical_30']

X_train['count_null'] = X_train.isnull().sum(axis = 1)
X_test['count_null'] = X_test.isnull().sum(axis = 1)

for col in cols_to_use_2:
  X_train[str(col + '_na')] = pd.isnull(X_train[col])
  X_test[str(col + '_na')] = pd.isnull(X_test[col])
mean_values = X_train.mean(axis = 0)

X_train.fillna(mean_values, inplace = True)
X_test.fillna(mean_values, inplace = True)

# for col in cols_to_use_2:
#   print col, X_train[col].max(), mean_values[col]


ymedian_dict = dict(X_train.groupby(["id"])["y"].median())
def get_weighted_y(id, y):
  return 0.95 * y + 0.05 * ymedian_dict[id] if id in ymedian_dict else y

output_file = open('output_id_wise.txt', 'w')

print ('training model, random_state is : ' + str(0.1))
model_1 = Ridge(alpha = 0.1)
model_1.fit(np.array(X_train.loc[y_is_within_cut, cols_to_use].values), X_train.loc[y_is_within_cut, 'y'])

model_2 = ExtraTreesRegressor(n_estimators = 100, max_depth = 4, n_jobs = -1, random_state = 10000000)
model_2.fit(np.array(X_train.loc[y_is_within_cut, cols_to_use_2].values), X_train.loc[y_is_within_cut, 'y'])

print ('calculating score on entire test set')
output_file.write('calculating score on entire test set\n')
y_predicted = 0.8 * model_1.predict(X_test[cols_to_use]).clip(low_y_cut, high_y_cut) + 0.2 * model_2.predict(X_test[cols_to_use_2]).clip(low_y_cut, high_y_cut)
for i in range(len(y_predicted)):
  y_predicted[i] = get_weighted_y(int(X_test.iloc[i]['id'].values), float(y_predicted[i].values))

print ('score : ', np.correlate(y_test['y'], y_predicted))
output_file.write(str('score : ' + str(np.correlate(X_test['y'], y_predicted)) + '\n'))

output_file.close()
print ('done!')