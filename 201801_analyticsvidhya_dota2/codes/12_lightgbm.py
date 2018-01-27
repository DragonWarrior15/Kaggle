import pandas as pd
import numpy as np
import os
import sys
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor, plot_tree, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

epsilon = 1e-07

with open('../inputs/hero_id_dict', 'rb') as f:
    hero_id_dict = pickle.load(f)

# read the inputs
df_heroes = pd.read_csv('../inputs/hero_data_processed.csv')
# correct the hero ids
df_heroes['hero_id'] = df_heroes['hero_id'].apply(lambda x: hero_id_dict[x])

df_train_full = []
for file in ['train9.csv', 'train1.csv', 'test9.csv']:
    df_temp = pd.read_csv('../inputs/' + file)
    df_train_full.append(df_temp.loc[:,:])

df_train_full = pd.concat(df_train_full)

# correct the hero ids in the dataset
df_train_full['hero_id'] = df_train_full['hero_id'].apply(lambda x: hero_id_dict[x])
# correct the user_id, decrease them by 1 so that user ids start from 0
df_train_full['user_id'] = df_train_full['user_id'].apply(lambda x: x - 1)

# prepare a dict to map column name to position
train_cols_dict = dict([[x, index] for index, x in enumerate(df_train_full.columns.tolist())])

# join train and heroes dataset on hero_id
df_train_full = pd.merge(left = df_train_full, right = df_heroes, how = 'inner', on = 'hero_id')
# df_train_full.drop(['num_wins'], axis = 1, inplace = True)

input_cols = [x for x in df_train_full.columns.tolist() if x != 'kda_ratio']
target_cols = ['kda_ratio']

model_list = []
seed_list = [100 * (x + 1) + 10 * (x + 2) + (x + 3) for x in range(10)]
for cross_val in range(5):
    # split the data into train and validation sets
    # validation set will comprise of a single rating removed from each
    # of the users
    df_val = df_train_full.sample(frac = 1, replace = False, random_state = seed_list[cross_val])
    df_val = df_val.drop_duplicates(subset = ['user_id'], keep = 'first')
    df_remove_from_train = pd.DataFrame(df_val['id'])
    df_remove_from_train['drop'] = 1
    df_train = pd.merge(left = df_train_full, right = df_remove_from_train, how = 'left', on = 'id')
    df_train = df_train.loc[df_train['drop'] != 1, :]
    df_train.drop('drop', axis = 1, inplace = True)

    X_train, X_val = df_train[input_cols].as_matrix(),  df_val[input_cols].as_matrix()
    y_train, y_val = df_train[target_cols].as_matrix().reshape(-1), df_val[target_cols].as_matrix().reshape(-1)
    
    reg = LGBMRegressor(random_seed=100, learning_rate = 0.1, n_estimators = 1000, \
                             max_depth = 4, colsample_bytree = 0.8, reg_alpha = 0.1,\
                             min_child_weight = 2, subsample = 0.95, subsample_for_bin = 10)
        
    reg.fit(X_train, y_train)
    model_list.append(reg)
    
    print(str(cross_val) + ',' + ' Train : ' + str(round(mean_squared_error(y_train, reg.predict(X_train)) ** 0.5, 5)) + \
          ' , Val : ' + str(round(mean_squared_error(y_val, reg.predict(X_val)) ** 0.5, 5)))
