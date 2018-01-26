import pandas as pd
import numpy as np
import os
import sys
import pickle

epsilon = 1e-07
num_iterations = 100
learning_rate = 0.01

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
hero_matrix = (hero_matrix - np.mean(hero_matrix, axis = 0))/(np.std(hero_matrix, axis = 0) + epsilon)
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


# start the cross validation loop, use 10 cross validations
seed_list = [100 * (x + 1) + 10 * (x + 2) + (x + 3) for x in range(10)]
train_history = {'train_loss':[], 'val_loss':[]}
for cross_val in range(1):
    # define two numpy arrays, one stores 1 at the combinations of 
    # train user X hero, and other stores for val
    is_val = np.zeros((num_users, num_heroes))
    is_train = np.array(user_hero_exist)

    # define the attribute matrix, num of columns is same as no of 
    # hero attributes, add a bias term also
    # user_matrix = np.random.randn(num_users, num_hero_attr)
    user_matrix = np.zeros((num_users, num_hero_attr))

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
    num_games_std  = np.std(is_train_num_games_list)
    # standard scale those values
    user_hero_num_games_train = (user_hero_num_games - num_games_mean)/num_games_std

    ## begin the training loop
    for iteration in range(num_iterations):
        # calculate the cost
        cost = np.matmul(user_matrix, hero_matrix.T) + (user_hero_num_games_train)
        cost_cache = np.multiply(is_train, cost) - np.multiply(is_train, user_hero_kda)
        cost = np.power(np.sum(cost), 2)/(2 * np.sum(is_train))
        # update the train history
        train_history['train_loss'].append(cost)

        ## update the weights
        # loop over all the weights in the matrix
        user_matrix -= (learning_rate * np.matmul(np.multiply(is_train, cost_cache), hero_matrix))/np.sum(is_train)

        ## little more detailed logic using for loos
        # for row in range(user_matrix.shape[0]):
            # for col in range(user_matrix.shape[1]):
                # gradient = 0
                # loop over all the heros for which cost was calculated
                # gradient += cost_cache[row][col] * np.multiply(is_train[row, :], hero_matrix.T[col, :])
                # update the weights
                # user_matrix[row][col] -= (learning_rate * gradient)/np.sum(is_train)

        # calculate the validation loss
        cost = np.matmul(user_matrix, hero_matrix.T) + (user_hero_num_games_train)
        cost_cache = np.multiply(is_val, cost) - np.multiply(is_val, user_hero_kda)
        cost = np.power(np.sum(cost), 2)/(2 * np.sum(is_val))
        # update the train_history
        train_history['val_loss'].append(cost)

        print (cross_val, iteration, train_history['train_loss'][-1], train_history['val_loss'][-1])
