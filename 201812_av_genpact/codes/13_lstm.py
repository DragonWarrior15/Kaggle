import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, SimpleRNN
from keras.preprocessing import sequence

# time steps to look back
look_back = 3
feature_list = ['center_id', 'meal_id', 'base_price', 'checkout_price', 'num_orders', 
                'emailer_for_promotion', 'homepage_featured', 'city_code', 'region_code']
prepare_train_data = False
do_train = False
do_test = True

def rmse(y_true, y_pred):
    return(np.sqrt(mean_squared_error(y_true, y_pred)))

def get_curr_dt():
    curr = datetime.now()
    dt = [0, 0, 0, 0, 0]
    dt[0] = str(curr.year)
    dt[1] = curr.month
    dt[2] = curr.day
    dt[3] = curr.hour
    dt[4] = curr.minute
    # dt[5] = curr.second

    for i in range(1, len(dt)):
        if(dt[i]//10) == 0:
            dt[i] = '0' + str(dt[i])
        else:
            dt[i] = str(dt[i])
        if(i == 3):
            dt[i] = '_' + dt[i]

    return(''.join(dt))

def prepare_features(df, min_week = 1, max_week = 0, verbose = 0):
    if(min_week <= look_back):
        min_week = look_back
    if(max_week - min_week < 0):
        max_week = df['week'].max()
    # filter data as needed
    df = df.loc[(df['week'] >= min_week - look_back) & (df['week'] <= max_week), :]
    # make the log scale
    df['num_orders'] = np.log(df['num_orders'])
    # list of all weeks
    week_list = list(range(min(df['week']), max(df['week']) + 1))
    # preprocess to the shape (center X meal) X (week) X (features - indicators, discount etc)
    X_train = []
    y_train = []
    print('Preparing Data ', datetime.now())
    curr_center = 0
    for comb, x in df.groupby(['center_id', 'meal_id']):
        if(comb[0] > curr_center):
            curr_center = comb[0]
            if(verbose != 0):
                print('Current Center ', curr_center)

        # to get all weeks for all center X meals, can have empty entries
        df_temp = pd.DataFrame({'week':week_list})
        df_temp['center_id'] = comb[0]
        df_temp['meal_id'] = comb[1]

        X = pd.merge(left = df_temp, right = x, how = 'left', on = ['week', 'center_id', 'meal_id'])
        X = X[feature_list]
        X.fillna(-1, inplace = True)

        X = X.as_matrix()
        for i in range(look_back + 1):
            if(i == look_back):
                # add target week features
                # make the num_orders null
                y_temp = X[look_back:, feature_list.index('num_orders')].copy()
                y_train.append(y_temp)
                X[:, feature_list.index('num_orders')] = -1
            try:
                X_temp2 = X[i : X.shape[0]-look_back+i].copy()
                X_temp2 = X_temp2[:, :, np.newaxis]
                X_temp2 = X_temp2.reshape(X_temp2.shape[0], 1, X_temp2.shape[1])
                # print(X_temp.shape, X_temp2.shape)
                X_temp = np.concatenate([X_temp, X_temp2], axis = 1)
            except NameError:
                # we want the input array to be samples X time steps X feature length
                # at every time step, we select some examples from the training data
                # this corresponds to some time step for training
                X_temp = X[i : X.shape[0]-look_back+i].copy()
                # reshape to 3d
                X_temp = X_temp[:, :, np.newaxis]
                # reshape to samples X 1 X features, and append new time steps
                # at the columns to get relevant shape
                X_temp = X_temp.reshape(X_temp.shape[0], 1, X_temp.shape[1])

        X_train.append(X_temp.copy())
        # get the targets
        
        del X_temp  

    print('Data Prepared ', datetime.now())
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train).reshape(-1)

    print(X_train.shape, y_train.shape)

    del df_temp
    del X
    # del X_temp
    gc.collect()

    return(X_train, y_train)


if(prepare_train_data == True):
    # read train data
    df = pd.read_csv('../inputs/train.csv')
    df_center = pd.read_csv('../inputs/fulfilment_center_info.csv')
    df = pd.merge(left = df, right = df_center, how = 'left', on = 'center_id')
    df = df.sort_values(['week', 'center_id', 'meal_id'], ascending = True)

    X, y = prepare_features(df, verbose = 1)

    with open('../inputs/X_%d' % (look_back), 'wb') as f:
        pickle.dump(X, f)
    with open('../inputs/y_%d' % (look_back), 'wb') as f:
        pickle.dump(y, f)    


if(do_train == True):
    # read data
    with open('../inputs/X_%d' % (look_back), 'rb') as f:
        X = pickle.load(f)
    with open('../inputs/y_%d' % (look_back), 'rb') as f:
        y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


    # start building model
    model = Sequential()
    model.add(SimpleRNN(1, input_shape=(look_back + 1, len(feature_list)), activation = 'linear'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=5, batch_size=141, verbose=1,
              validation_data = (X_test, y_test))

    model.save('../models/simple_rnn')

    print('Train RMSE : ', rmse(model.predict(X_train), y_train))
    print('Test RMSE : ', rmse(model.predict(X_test), y_test))


# sanity check
if(False):
    model = load_model('../models/simple_rnn')
    with open('../inputs/X_%d' % (look_back), 'rb') as f:
        X = pickle.load(f)
    with open('../inputs/y_%d' % (look_back), 'rb') as f:
        y = pickle.load(f)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 20)

    print(y.sum(), y.shape)
    print(y_test, model.predict(X_test))
    

# prepare test dataset features and
# make predictions on it (rolling needed)
if(do_test == True):
    # read train data
    df = pd.read_csv('../inputs/train.csv')
    df_test = pd.read_csv('../inputs/test.csv')
    df_test['num_orders'] = 1
    df = pd.concat([df, df_test])
    df = df.loc[df['week'] >= 130]
    df_center = pd.read_csv('../inputs/fulfilment_center_info.csv')
    df = pd.merge(left = df, right = df_center, how = 'left', on = 'center_id')
    df = df.sort_values(['week', 'center_id', 'meal_id'], ascending = True)

    model = load_model('../models/simple_rnn')

    test_weeks = [min(df_test['week']), max(df_test['week'])]
    predictions = []
    for i in range(test_weeks[0], test_weeks[0] + 2):
        X, y = prepare_features(df, min_week = i-1, max_week = i-1)
        y = model.predict(X)
        df_pred = pd.DataFrame({'center_id' : X[:, feature_list.index('center_id')],
                                'meal_id' : X[:, feature_list.index('meal_id')],
                                'num_orders' : y[0]})
        df_pred['week'] = i
        print(df_pred)
        predictions.append(df_test.copy())
        # modify the predictions in the test set as well
        df_test = pd.merge(left = df_test, right = df_pred, how = 'left',
                           on = ['week', 'center_id', 'meal_id'], suffixes = ('', '_model'))
        df_test.loc[~df_test['num_orders_model'].isnull(), 'num_orders'] = df_test.loc[~df_test['num_orders_model'].isnull(), 'num_orders_model']
        df_test.drop(['num_orders_model'], axis = 1, inplace = True)

    predictions = pd.concat(predictions)

    predictions = pd.merge(left = df_test, right = predictions, on = ['week', 'center_id', 'meal_id'],
                           how = 'left')
    predictions = predictions[['id', 'num_orders']]
    predictions.to_csv('../submissions/lstm_%s.csv' % (get_curr_dt()), index = False)
    
