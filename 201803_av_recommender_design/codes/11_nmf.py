import pandas as pd
import numpy as np
import os
import sys
import pickle

import map_at_k as eval_metric

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import train_test_split

from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy import sparse
from scipy import linalg
from numpy import dot

# from lightgbm import LGBMClassifier, LGBMRegressor
# from keras.models import Sequential
# from keras.layers import Dense

epsilon = 1e-07

predict_on_test = True
cross_val_start = 6
num_cross_val = 5

def nmf(V, k, threshold=1e-5, maxiter=100):
    
    d, n = V.shape
    W = np.random.rand(d, k)
    H = np.random.rand(k, n)
    WH = np.dot(W, H)
    F = (V * np.log(WH) - WH).sum() / (d * n)
    
    it_no = 0
    converged = False

    while (not converged) and it_no <= maxiter:
        WH = np.dot(W, H)
        W_new = W * np.dot(V / WH, H.T)
        W_new = W_new / np.sum(W_new, axis=0, keepdims=True)
        H_new = H * np.dot((V / WH).T, W).T
        F_new = (V * np.log(WH) - WH).sum() / (d * n)

        converged = np.abs(F_new - F) <= threshold 
        W, H = W_new, H_new
        it_no = it_no + 1
    
    return W, H

df_train_input = pd.read_csv('../inputs/train.csv')
df_train_input = df_train_input.drop(['user_sequence'], axis = 1)

model_list = []
# start the cross validation loop, use 10 cross validations
seed_list = [100 * (x + 1) + 10 * (x + 2) + (x + 3) for x in range(100)]
for num_components in [5, 8, 10]:
    first_cross_val = True
    for cross_val in range(cross_val_start, num_cross_val + cross_val_start):
        ## split the train data into train and validation
        df_train, df_val = train_test_split(df_train_input, test_size = 0.2, random_state = seed_list[cross_val])

        ## split the validation data so that last 3 challenges are not
        ## used for preparing the matrix
        df_val_first10 = df_val.loc[df_val['challenge_sequence'] <= 10, :]
        df_val_last3 = df_val.loc[df_val['challenge_sequence'] > 10, :]
        df_val_last3 = df_val_last3.sort_values(by = ['user_id', 'challenge'])

        ## use both training, and first 10 challenges of validation data for training
        df_train = pd.concat([df_train, df_val_first10])
        df_train = df_train.drop(['challenge_sequence'], axis = 1)
        df_train['challenge_done'] = 1
        df_train = df_train.sort_values(by = ['user_id', 'challenge'])

        ## extract a user list and challenge list, so that they can
        ## be directly referred to using index
        user_list = sorted(df_train['user_id'].unique().tolist(), reverse = True)
        user_dict = dict([x, index] for index, x in enumerate(user_list))

        challenge_list = sorted(df_train['challenge'].unique().tolist(), reverse = True)
        challenge_dict = dict([x, index] for index, x in enumerate(challenge_list))

        ## craete the training matrix
        # W = df_train.pivot(index='user_id', columns='challenge', values='challenge_done').as_matrix()
        data = df_train['challenge_done'].tolist()
        row = df_train['user_id'].apply(lambda x: user_dict[x]).tolist()
        col = df_train['challenge'].apply(lambda x: challenge_dict[x]).tolist()
        W = csr_matrix((data, (row, col)), shape=(len(user_list), len(challenge_list)))

        ## train the model
        model = NMF(num_components)
        model.fit(W)
        V = model.components_
        W = model.transform(W)

        ## get the predictions for the users in val data
        user_list_val = df_val['user_id'].unique().tolist()

        ## prepare a predicted and actual list for evaluation
        predicted_list = []
        actual_list = []

        for user in user_list_val:
            ## get the predictions from W for this user
            predictions_for_user = list(W[user_dict[user], :].reshape(1, -1))
            ## assign challenges to this list
            predictions_for_user = zip(challenge_list, predictions_for_user)
            ## remove the challenges already done by user
            challenges_already_done_by_user = df_val_first10.loc[df_val_first10['user_id'] == user, 'challenge'].tolist()
            predictions_for_user = [x for x in predictions_for_user if x[0] not in challenges_already_done_by_user]
            ## sort the challenges
            predictions_for_user = sorted(predictions_for_user, key = lambda x: x[1], reverse = False)
            predictions_for_user = predictions_for_user[:3]

            predicted_list.append([x[0] for x in predictions_for_user])
            actual_list.append(df_val_last3.loc[df_val_last3['user_id'] == user, 'challenge'].tolist())

        ## get the metrics
        map_k = eval_metric.mapk(actual_list, predicted_list, k = 3)
        print ('Num Components : %02d, Cross Val Num : %02d, MAP : %.5f' %(num_components, cross_val, map_k))

'''
Num Components : 05, Cross Val Num : 06, MAP : 0.00001
Num Components : 05, Cross Val Num : 07, MAP : 0.00000
Num Components : 05, Cross Val Num : 08, MAP : 0.00000
Num Components : 05, Cross Val Num : 09, MAP : 0.00002
Num Components : 05, Cross Val Num : 10, MAP : 0.00000
Num Components : 08, Cross Val Num : 06, MAP : 0.00001
Num Components : 08, Cross Val Num : 07, MAP : 0.00000
Num Components : 08, Cross Val Num : 08, MAP : 0.00000
Num Components : 08, Cross Val Num : 09, MAP : 0.00002
Num Components : 08, Cross Val Num : 10, MAP : 0.00000
Num Components : 10, Cross Val Num : 06, MAP : 0.00001
Num Components : 10, Cross Val Num : 07, MAP : 0.00000
Num Components : 10, Cross Val Num : 08, MAP : 0.00000
Num Components : 10, Cross Val Num : 09, MAP : 0.00002
Num Components : 10, Cross Val Num : 10, MAP : 0.00000
'''
