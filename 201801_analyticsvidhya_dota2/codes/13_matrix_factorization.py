import pandas as pd
import numpy as np
import os
import sys
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds

epsilon = 1e-07
num_iterations = 2000
learning_rate = 1
reg_alpha = 1
n_components_pca = 2
base_kda = 3000

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

# get the unique users
num_users = len(df_train_full['user_id'].unique())
# add a term for bias, remove the hero id
# num_hero_attr = len(df_heroes.columns.tolist()) - 1 + 1
num_hero_attr = n_components_pca + 1 # changed to use pca
num_heroes = len(df_heroes['hero_id'].unique())

df_test = pd.read_csv('../inputs/test1.csv')
df_test['user_id'] = df_test['user_id'].apply(lambda x: x - 1)
df_test['hero_id'] = df_test['hero_id'].apply(lambda x: hero_id_dict[x])

# the idea is to derive user attributes/coefficients for all the users
# to make things familiar, imagine the system as different users, who have rated
# different movies (rating is kda and movie is hero) Now using the ratings, we need
# to identify every customer's attribute vector, each attribute provides information
# like how much regen does the user prefer, how much weight to mana etc

## preprocessing
df_heroes.drop(['hero_id'], axis = 1, inplace = True)
hero_matrix = df_heroes.as_matrix()
pca = PCA(n_components_pca)
pca.fit(hero_matrix)
hero_matrix = pca.transform(hero_matrix)
print (pca.explained_variance_ratio_)
# standard scale the hero_matrix
hero_matrix = (hero_matrix - np.mean(hero_matrix, axis = 0))/(np.std(hero_matrix, axis = 0) + epsilon)
hero_matrix = np.hstack([np.ones((hero_matrix.shape[0], 1)), hero_matrix])
# hero_matrix[:, 0] = 1
# print (hero_matrix)

# create a matrix to store the target vaiable for
# different user X hero pairs
user_hero_kda_raw   = df_train_full.pivot_table(index = 'user_id', columns = 'hero_id', values = 'kda_ratio').fillna(0).as_matrix()
user_hero_num_games = df_train_full.pivot_table(index = 'user_id', columns = 'hero_id', values = 'num_games').fillna(0).as_matrix()
user_hero_num_wins  = df_train_full.pivot_table(index = 'user_id', columns = 'hero_id', values = 'num_wins').fillna(0).as_matrix()

user_hero_exist = np.clip(np.array(user_hero_num_games), a_min = 0, a_max = 1)

model_list = []
# start the cross validation loop, use 10 cross validations
seed_list = [100 * (x + 1) + 10 * (x + 2) + (x + 3) for x in range(100)]
for cross_val in range(1, 21):
    # define two numpy arrays, one stores 1 at the combinations of 
    # train user X hero, and other stores for val
    is_val   = np.zeros((num_users, num_heroes))
    is_train = np.array(user_hero_exist)
    # split the data into train and validation sets
    # validation set will comprise of a single rating removed from each
    # of the users
    df_val = df_train_full.sample(frac = 1, replace = False, random_state = seed_list[cross_val])
    df_val = df_val.drop_duplicates(subset = ['user_id'], keep = 'first')
    for i in range(len(df_val)):
        row = df_val.iloc[i, train_cols_dict['user_id']]
        col = df_val.iloc[i, train_cols_dict['hero_id']]
        is_val[row][col] = 1
        is_train[row][col] = 0

    user_hero_kda  = np.array(user_hero_kda_raw)
    # user_hero_kda  = (user_hero_kda_raw/np.clip(user_hero_num_wins, a_min = 1, a_max = None))
    num_games_mean = np.mean(user_hero_num_games[is_train == 1])
    num_wins_mean  = np.mean(user_hero_num_wins[is_train == 1])
    kda_mean       = np.mean(np.multiply(is_train, user_hero_kda), axis = 0)
    kda_std        = np.std(np.multiply(is_train, user_hero_kda), axis = 0) + epsilon
    user_hero_kda  = (user_hero_kda - kda_mean)/kda_std

    user_matrix, weight_matrix, hero_feature_matrix =  svds(np.multiply(is_train, user_hero_kda), k = 10)
    model_list.append({'user_matrix':user_matrix, 'weight_matrix':weight_matrix, 'hero_feature_matrix':hero_feature_matrix,\
                       'kda_mean':kda_mean, 'kda_std':kda_std})
    # print (weight_matrix)
    weight_matrix = np.diag(weight_matrix)
    # print (user_matrix.shape, weight_matrix.shape, hero_feature_matrix.shape)

    # make predictions on the all the combinations
    kda_pred   = np.matmul(np.matmul(user_matrix, weight_matrix), hero_feature_matrix)
    error      = kda_pred - user_hero_kda
    cost_train = np.sqrt(np.sum(np.square(np.multiply(is_train, error)))/np.sum(is_train))
    cost_val   = np.sqrt(np.sum(np.square(np.multiply(is_val, error)))/np.sum(is_val))
    # cost_val_actual = np.sqrt((np.sum(np.square(np.multiply(is_val, user_hero_kda_raw - np.multiply(kda_pred * kda_std + kda_mean, user_hero_num_games) + base_kda))))/np.sum(is_val))
    cost_val_actual = np.sqrt((np.sum(np.square(np.multiply(is_val, user_hero_kda_raw - (kda_pred * kda_std + kda_mean)))))/np.sum(is_val))
    
    print ('Val Iter : ' + str(cross_val) +  ', Train : ' + str(round(cost_train, 2)) + ', Val : ' + str(round(cost_val, 2)) + \
           ', Actual Val : ' + str(round(cost_val_actual, 2)))
        
'''
pd.DataFrame(kda_pred).to_csv('../graphs/v1.csv')
pd.DataFrame(user_hero_num_games).to_csv('../graphs/v2.csv')
pd.DataFrame(user_hero_kda_raw).to_csv('../graphs/v3.csv')
print (kda_mean, kda_std)
'''

sys.exit()
# make the predictions on the test set
df_test['kda_ratio'] = kda_mean
kda_pred_np = sess.run(kda_pred)
user_matrix_num_games_np = sess.run(user_matrix_num_games)
for model in model_list:
    kda_pred = np.matmul(np.matmul(model['user_matrix'], model['weight_matrix'], model['hero_feature_matrix']))
    for i in range(len(df_test)):
        df_test.iloc[i, 4] += kda_pred[df_test.iloc[i, 0]][df_test.iloc[i, 1]] * df_test.iloc[i, 3] * model['kda_std'] + model['kda_mean']
df_test['kda_ratio'] /= len(model_list)
df_test[['id', 'kda_ratio']].to_csv('../submissions/submissions_20180127_1636_mf.csv', index=False)
