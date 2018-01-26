import pandas as pd
import numpy as np
import os
import sys
import pickle
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


epsilon = 1e-07
num_iterations = 3000
learning_rate = 0.001
reg_alpha = 0.5

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
num_hero_attr = len(df_heroes.columns.tolist()) - 1 + 1
num_heroes = len(df_heroes['hero_id'].unique())

# the idea is to derive user attributes/coefficients for all the users
# to make things familiar, imagine the system as different users, who have rated
# different movies (rating is kda and movie is hero) Now using the ratings, we need
# to identify every customer's attribute vector, each attribute provides information
# like how much regen does the user prefer, how much weight to mana etc

## preprocessing
# change the hero_id so that it serves as the bias term
df_heroes['hero_id'] = 1
hero_matrix = df_heroes.as_matrix()
# standard scale the hero_matrix
# hero_matrix = (hero_matrix - np.mean(hero_matrix, axis = 0))/(np.std(hero_matrix, axis = 0) + epsilon)
hero_matrix[:, 0] = 1
# print (hero_matrix)

# create a matrix which stores the num of games played
# using the hero
user_hero_num_games = np.zeros((num_users, num_heroes))
# create a matrix to store the target vaiable for
# different user X hero pairs
user_hero_kda = df_train_full.pivot_table(index = 'user_id', columns = 'hero_id', values = 'kda_ratio').fillna(0).as_matrix()
user_hero_num_games = df_train_full.pivot_table(index = 'user_id', columns = 'hero_id', values = 'num_games').fillna(0).as_matrix()

user_hero_exist = np.clip(np.array(user_hero_num_games), a_min = 0, a_max = 1)

hero_matrix   = tf.constant(hero_matrix  , dtype = tf.float32)
user_hero_kda = tf.constant(user_hero_kda, dtype = tf.float32)

# start the cross validation loop, use 10 cross validations
seed_list = [100 * (x + 1) + 10 * (x + 2) + (x + 3) for x in range(10)]
for cross_val in range(5):
    train_history = {'train_loss':[], 'val_loss':[]}
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
    ## scale and transform the num_games values
    # get all the list of num_games
    is_train_num_games_list = user_hero_num_games[is_train == 1]
    num_games_mean = np.mean(is_train_num_games_list)
    num_games_std  = np.std(is_train_num_games_list) + epsilon
    
    # standard scale those values
    user_hero_num_games_train = tf.constant((user_hero_num_games - num_games_mean)/num_games_std, dtype = tf.float32)

    is_val   = tf.constant(is_val  , dtype = tf.float32)
    is_train = tf.constant(is_train, dtype = tf.float32)


    # define the attribute matrix, num of columns is same as no of 
    # hero attributes, add a bias term also
    user_matrix           = tf.Variable(tf.zeros([num_users, num_hero_attr]), dtype = tf.float32)
    user_matrix_num_games = tf.Variable(tf.zeros([num_users, 1])            , dtype = tf.float32)

    kda_pred      = tf.matmul(user_matrix, tf.transpose(hero_matrix)) + tf.multiply(user_matrix_num_games, user_hero_num_games_train)
    # kda_pred      = tf.matmul(user_matrix, tf.transpose(hero_matrix))
    error         = kda_pred - user_hero_kda
    cost          = tf.reduce_sum(tf.square(tf.multiply(is_train, error)))/tf.reduce_sum(is_train)
    cost_training = cost + reg_alpha * (tf.reduce_sum(tf.square(user_matrix) + tf.square(user_matrix_num_games)))/tf.reduce_sum(is_train)
    cost_train    = tf.sqrt(cost)
    cost_val      = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply(is_val, error)))/tf.reduce_sum(is_val))
    
    # train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cost_training)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_training)
    # train_step  = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_training)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for iteration in range(num_iterations):
        sess.run(train_step)
        train_history['train_loss'].append(sess.run(cost_train))
        train_history['val_loss'].append(sess.run(cost_val))

        if (iteration + 1)% 500 == 0:
            print (cross_val, iteration, train_history['train_loss'][-1], train_history['val_loss'][-1])

    with sns.plotting_context(font_scale=1.3):
        fig, axs = plt.subplots(nrows = 1, figsize=(18, 8))
        sns.set(font_scale=1.5)
        sns.set_style("darkgrid")
        plt.plot(train_history['train_loss'], label = 'Train'     , color = 'b', linewidth = 1)
        plt.plot(train_history['val_loss']  , label = 'Validation', color = 'r', linewidth = 2)
        legend = axs.legend(loc='upper right')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        fig.savefig('../graphs/collaborative_filtering_tf_v1_cross_val_' + str(cross_val) + '.png', dpi = 300)
