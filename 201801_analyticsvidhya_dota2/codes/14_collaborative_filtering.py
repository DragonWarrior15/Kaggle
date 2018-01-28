import pandas as pd
import numpy as np
import os
import sys
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, NMF
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from scipy.sparse.linalg import svds
from scipy import sparse
from scipy import linalg
from numpy import dot

# from keras.models import Sequential
# from keras.layers import Dense

epsilon = 1e-07
num_iterations_user_mat = 1000
learning_rate_user_mat = 10
alpha_user_mat = 1
num_iterations_nmf = 1000
learning_rate_nmf = 0.01
alpha_nmf = 1
reg_alpha = 1
n_components_pca = 2
n_components_nmf = 2
base_kda = 3000

with open('../inputs/hero_id_dict', 'rb') as f:
    hero_id_dict = pickle.load(f)

# read the inputs
df_heroes = pd.read_csv('../inputs/hero_data_processed.csv')
# correct the hero ids
df_heroes['hero_id'] = df_heroes['hero_id'].apply(lambda x: hero_id_dict[x])
hero_matrix = df_heroes.as_matrix()
pca = PCA(n_components_pca)
pca.fit(hero_matrix)
hero_matrix = pca.transform(hero_matrix)
# print (pca.explained_variance_ratio_)
# standard scale the hero_matrix
hero_matrix = (hero_matrix - np.mean(hero_matrix, axis = 0))/(np.std(hero_matrix, axis = 0) + epsilon)
hero_matrix = np.hstack([np.ones((hero_matrix.shape[0], 1)), hero_matrix])

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
num_heroes = len(df_heroes['hero_id'].unique())

num_hero_attr = n_components_pca + 1 # changed to use pca

# create a matrix to store the target vaiable for
# different user X hero pairs
p1 = np.percentile(df_train_full['kda_ratio'], 1)
p99 = np.percentile(df_train_full['kda_ratio'], 99)
user_hero_kda_raw       = df_train_full.pivot_table(index = 'user_id', columns = 'hero_id', values = 'kda_ratio').fillna(0).as_matrix()
user_hero_kda_raw       = np.clip(user_hero_kda_raw, a_min = p1, a_max = p99)
user_hero_num_games_raw = df_train_full.pivot_table(index = 'user_id', columns = 'hero_id', values = 'num_games').fillna(0).as_matrix()
user_hero_num_wins_raw  = df_train_full.pivot_table(index = 'user_id', columns = 'hero_id', values = 'num_wins').fillna(0).as_matrix()

user_hero_exist = np.clip(np.array(user_hero_num_games_raw), a_min = 0, a_max = 1)

model_list = []
# start the cross validation loop, use 10 cross validations
seed_list = [100 * (x + 1) + 10 * (x + 2) + (x + 3) for x in range(100)]
for cross_val in range(6,7):
    # define two numpy arrays, one stores 1 at the combinations of 
    # train user X hero, and other stores for val
    is_val   = np.zeros((num_users, num_heroes))
    is_train = np.array(user_hero_exist)
    # split the data into train and validation sets
    # validation set will comprise of a single rating removed from each
    # of the users
    df_val = df_train_full.sample(frac = 1, replace = False, random_state = seed_list[cross_val])
    df_val = df_val.drop_duplicates(subset = ['user_id'], keep = 'first')
    df_val['kda_ratio_pred'] = 3000
    df_val['user_mean_kda'] = 3000
    df_val['hero_mean_kda'] = 3000
    df_val['kda_user_matrix'] = 3000
    df_val['kda_nmf'] = 3000
    # df_val['kda_svd'] = 3000


    for i in range(len(df_val)):
        row = df_val.iloc[i, train_cols_dict['user_id']]
        col = df_val.iloc[i, train_cols_dict['hero_id']]
        is_val[row][col] = 1
        is_train[row][col] = 0
    
    # use the similar users (based on cosine similarity from the kda matrix) to make the predictions
    # first normalize the kda + num_games + num_wins matrix
    user_hero_kda    = np.multiply(is_train, user_hero_kda_raw)
    kda_mean         = (np.sum(user_hero_kda, axis = 1)/np.sum(is_train, axis = 1)).reshape(-1, 1)
    user_hero_kda    = np.multiply(is_train, user_hero_kda - kda_mean)
    # user_hero_kda_l2 = np.sqrt(np.sum(np.square(user_hero_kda), axis = 1)).reshape(-1, 1)

    user_hero_num_games = np.multiply(is_train, user_hero_num_games_raw)
    num_games_mean      = (np.sum(user_hero_num_games, axis = 1)/np.sum(is_train, axis = 1)).reshape(-1, 1)
    user_hero_num_games = np.multiply(is_train, user_hero_num_games - num_games_mean)

    user_hero_num_wins = np.multiply(is_train, user_hero_num_wins_raw)
    num_wins_mean      = (np.sum(user_hero_num_wins, axis = 1)/np.sum(is_train, axis = 1)).reshape(-1, 1)
    user_hero_num_wins = np.multiply(is_train, user_hero_num_wins - num_wins_mean)
    
    # cosine_matrix    = np.hstack((user_hero_kda, user_hero_num_games, user_hero_num_wins))
    cosine_matrix    = np.array(user_hero_kda)
    cosine_matrix_l2 = np.sqrt(np.sum(np.square(cosine_matrix), axis = 1)).reshape(-1, 1)

    # get kda mean for every hero
    kda_hero_mean = (np.sum(np.multiply(user_hero_kda_raw, is_train).T, axis = 1)/np.sum(is_train.T, axis = 1)).reshape(-1, 1)
    # print (kda_hero_mean)

    # try to prepare a use matrix using the available hero attributes and kda ratios
    user_matrix = np.random.randn(num_users, hero_matrix.shape[1])
    for iteration in range(num_iterations_user_mat):
        kda_user_matrix_pred = np.matmul(user_matrix, hero_matrix.T)
        cost_cache = np.multiply(is_train, kda_user_matrix_pred - user_hero_kda_raw)
        user_matrix -= (learning_rate_user_mat/np.sum(is_train)) * (np.matmul(cost_cache, hero_matrix) + alpha_user_mat * user_matrix)

        cost = np.sum(np.square(np.multiply(is_train, cost_cache)))/(2 * np.sum(is_train))
        # if iteration%50 == 0:
            # print (cost, np.sqrt(cost))

    # user_matrix = np.linalg.lstsq(hero_matrix, user_hero_kda_raw.T)[0].T
    cost = np.sum(np.square(np.multiply(is_train, cost_cache)))/(2 * np.sum(is_train))
    kda_user_matrix_pred = np.matmul(user_matrix, hero_matrix.T)
    print ('user_matrix', cost, np.sqrt(cost))
    # extract user behaviour for kda using nmf
    nmf_user_matrix = np.random.randn(num_users, n_components_nmf)
    nmf_hero_matrix = np.random.randn(num_heroes, n_components_nmf)
    for iteration in range(num_iterations_nmf):
        kda_nmf_pred = np.matmul(nmf_user_matrix, nmf_hero_matrix.T)
        cost_cache = np.multiply(is_train, kda_nmf_pred - user_hero_kda_raw)
        nmf_user_matrix -= (learning_rate_nmf/np.sum(is_train)) * (np.matmul(cost_cache, nmf_hero_matrix) + alpha_nmf * nmf_user_matrix)
        nmf_hero_matrix -= (learning_rate_nmf/np.sum(is_train)) * (np.matmul(cost_cache.T, nmf_user_matrix) + alpha_nmf * nmf_hero_matrix)

        cost = np.sum(np.square(np.multiply(is_train, cost_cache)))/(2 * np.sum(is_train))
        # if iteration%50 == 0:
            # print (cost, np.sqrt(cost))
    
    cost = np.sum(np.square(np.multiply(is_train, cost_cache)))/(2 * np.sum(is_train))
    kda_nmf_pred = np.matmul(nmf_user_matrix, nmf_hero_matrix.T)
    print ('nmf', cost, np.sqrt(cost))

    '''
    # extract user behaviour using svd
    user_matrix_svd, weight_matrix, hero_feature_matrix =  svds(np.multiply(is_train, user_hero_kda_raw), k = 2)
    kda_svd_pred = np.matmul(np.matmul(user_matrix_svd, np.diag(weight_matrix)), hero_feature_matrix)
    cost = np.sum(np.square(np.multiply(is_train, (kda_svd_pred - user_hero_kda_raw))))/(2 * np.sum(is_train))
    print ('svd', cost, np.sqrt(cost))
    '''

    error = 0
    # for i in range(10):
    for i in range(len(df_val)):
        current_user = df_val.iloc[i, 0]
        current_hero = df_val.iloc[i, 1]
        # print ('current user is %d and current hero is %d'% (current_user, current_hero))

        # user_hero_kda_wo_user = np.delete(user_hero_kda, (current_user), axis = 0)
        cosine_similarities = np.divide(np.sum(np.multiply(cosine_matrix[current_user, :].reshape(1, -1), cosine_matrix), axis = 1).reshape(-1, 1),\
                                        cosine_matrix_l2)/cosine_matrix_l2[current_user][0]

        cosine_similarities_w_kda = np.hstack((cosine_similarities, user_hero_kda_raw[:, current_hero].reshape(-1, 1)))
        cosine_similarities_w_kda = cosine_similarities_w_kda[cosine_similarities_w_kda[:, 1] > p1]
        cosine_similarities_w_kda = cosine_similarities_w_kda[np.argsort(cosine_similarities_w_kda[:, 0]), :][-10:-1, :]


        df_val.iloc[i, 6] = np.sum(np.multiply(cosine_similarities_w_kda[:, 0], cosine_similarities_w_kda[:, 1]))/\
                            (np.sum(cosine_similarities_w_kda[:, 0]) + epsilon)
        
        # store the means at user and group levels
        df_val.iloc[i, 7]  = kda_mean[current_user]
        df_val.iloc[i, 8]  = kda_hero_mean[current_hero]
        df_val.iloc[i, 9]  = kda_user_matrix_pred[current_user][current_hero]
        df_val.iloc[i, 10] = kda_nmf_pred[current_user][current_hero]
        # df_val.iloc[i, 11] = kda_svd_pred[current_user][current_hero]

        error_curr = (df_val.iloc[i, 5] - df_val.iloc[i, 6])
        error += error_curr ** 2
        # print ('Currently Evaluated %4dth row and error is %4.2f, current actual kda %5d, pred kda %5d' % (i, error_curr, df_val.iloc[i, 5], df_val.iloc[i, 6]))

    cost = np.sqrt(error/len(df_val))
    # print ('Val Iter : ' + str(cross_val))
    print ('Val Iter : ' + str(cross_val) + ', Val : ' + str(round(cost, 2)))
    
    # fit a linear regression on the 
    linear_reg = LinearRegression()
    scaler = StandardScaler()
    X_train = df_val[['num_games', 'kda_ratio_pred', 'user_mean_kda', 'hero_mean_kda', 'kda_user_matrix', 'kda_nmf']].as_matrix()
    y_train = df_val['kda_ratio'].as_matrix()
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    linear_reg.fit(X_train, y_train)
    error = np.sqrt(mean_squared_error(linear_reg.predict(X_train), y_train))
    print ('linear_reg', error)
    # error = np.sqrt(mean_squared_error(linear_reg.predict(X_test), y_test))
    # print (error)
    # print (scaler.mean_)
    print (linear_reg.coef_, linear_reg.intercept_)
    df_val.to_csv('../inputs/temp.csv', index = False)

    '''
    # try to fit a simple neural network on the data
    X_train = scaler.fit_transform(X_train)
    model = Sequential()
    model.add(Dense(7, input_dim = 7, activation = 'relu'))
    model.add(Dense(14, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    model.fit(X_train, y_train, epochs = 100)
    error = np.sqrt(mean_squared_error(model.predict(X_train), y_train))
    print ('nn', error)
    '''

sys.exit()

# make the predictions on the test set
df_test = pd.read_csv('../inputs/test1.csv')
df_test['user_id'] = df_test['user_id'].apply(lambda x: x - 1)
df_test['hero_id'] = df_test['hero_id'].apply(lambda x: hero_id_dict[x])

df_test['kda_ratio'] = kda_mean
kda_pred_np = sess.run(kda_pred)
user_matrix_num_games_np = sess.run(user_matrix_num_games)
for model in model_list:
    kda_pred = np.matmul(np.matmul(model['user_matrix'], model['weight_matrix'], model['hero_feature_matrix']))
    for i in range(len(df_test)):
        df_test.iloc[i, 4] += kda_pred[df_test.iloc[i, 0]][df_test.iloc[i, 1]] * df_test.iloc[i, 3] * model['kda_std'] + model['kda_mean']
df_test['kda_ratio'] /= len(model_list)
df_test[['id', 'kda_ratio']].to_csv('../submissions/submissions_20180127_1636_mf.csv', index=False)
