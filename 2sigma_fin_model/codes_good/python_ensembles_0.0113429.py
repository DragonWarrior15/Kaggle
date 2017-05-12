import kagglegym
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from scipy.special import cbrt
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor

# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor

  
target = 'y'

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train

cols_to_use_2 = list(train.columns)
cols_to_use_2.remove('id')
cols_to_use_2.remove('y')

train['count_null'] = train.isnull().sum(axis = 1)

for col in cols_to_use_2:
  train[str(col + '_na')] = pd.isnull(train[col])
    
mean_values = train.mean(axis=0)

# train['technical_30'].clip(0, 0.013087218)
# train['technical_20'].clip(0, 0.013016532)
# train['fundamental_11'].clip(-6.270231009, 1.189538956)

# median_values = train.median(axis=0)
train.fillna(mean_values, inplace=True)

# cols_to_use = ['technical_20']
# cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

# print('preparing new features')

# cols_to_use = ['fundamental_55_sin', 'technical_20', 'technical_30']
# cols_to_use = ['fundamental_11', 'fundamental_55_sin', 'technical_20', 'technical_30']
cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

# low_y_cut = -0.0860941261053085
# high_y_cut = 0.078021827

y_is_above_cut = (train.y > high_y_cut)
y_is_below_cut = (train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)

# ymean_dict = dict(train.groupby(["id"])["y"].mean())
ymedian_dict = dict(train.groupby(["id"])["y"].median())
    
def get_weighted_y(series):
  id, y = series["id"], series["y"]
  # return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y
  return 0.95 * y + 0.05 * ymedian_dict[id] if id in ymedian_dict else y

# normalize the feature set
# norms = np.linalg.norm(train[cols_to_use], axis=0)
# train[cols_to_use] /= norms


# model = RandomForestRegressor(n_estimators=100)
# model = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2,
#                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3)

# model = Ridge(alpha = 0.1)
model_1 = Ridge(alpha = 0.1)
# model = LinearRegression()
model_1.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target])

model_2 = ExtraTreesRegressor(n_estimators = 100, max_depth = 4, n_jobs = -1, random_state = 10000000)
model_2.fit(np.array(train.loc[y_is_within_cut, cols_to_use_2].values), train.loc[y_is_within_cut, target])

# cols_to_use_3 = ['technical_20']
# model_3 = LinearRegression()
# model_3.fit(np.array(train.loc[y_is_within_cut, cols_to_use_3].values), train.loc[y_is_within_cut, target])
  
while True:
  observation.features['count_null'] = observation.features.isnull().sum(axis = 1)
  for col in cols_to_use_2:
    observation.features[str(col + '_na')] = pd.isnull(observation.features[col])
  observation.features.fillna(mean_values, inplace=True)

  # observation.features[cols_to_use] /= norms
  test_x = np.array(observation.features[cols_to_use].values)
  test_x_2 = np.array(observation.features[cols_to_use_2].values)
  
  # observation.target.y = (0.8 * model_1.predict(test_x).clip(low_y_cut, high_y_cut) + 0.1 * model_2.predict(test_x_2).clip(low_y_cut, high_y_cut) + 0.1 * model_2.predict(test_x_2).clip(low_y_cut, high_y_cut)).clip(low_y_cut, high_y_cut)
  observation.target.y = 0.8 * model_1.predict(test_x).clip(low_y_cut, high_y_cut) + 0.2 * model_2.predict(test_x_2).clip(low_y_cut, high_y_cut)
  # observation.target.y = max(model_1.predict(test_x).clip(low_y_cut, high_y_cut), model_2.predict(test_x_2).clip(low_y_cut, high_y_cut))
  ## weighted y using average value
  observation.target.y = observation.target.apply(get_weighted_y, axis = 1)
      
  target = observation.target
  timestamp = observation.features["timestamp"][0]
  if timestamp % 100 == 0:
    print("Timestamp #{}".format(timestamp))
    pass
      
  observation, reward, done, info = env.step(target)
  if done:
    break
      
print(info)

