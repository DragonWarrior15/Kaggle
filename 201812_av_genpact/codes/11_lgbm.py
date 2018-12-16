import pandas as pd
import numpy as np
from datetime import datetime
import gc
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from lightgbm import LGBMRegressor

def rmse(y_true, y_pred):
    return(np.sqrt(mean_squared_error(y_true, y_pred)))

def se(y_true, y_pred):
    return(np.sum(np.power(y_true - y_pred, 2)))

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

def grid_search_sample_weight(model, X, y, \
                              search_params_dict, \
                              scoring = 'mean_squared_error', cv = 3, \
                              sample_weight_list = [1]):
    # code for grid search
    cv_df_list = []
    for sample_weight_coef in sample_weight_list:
        grid_search = GridSearchCV(model, search_params_dict, scoring = scoring, cv = cv)
        grid_search.fit(X, y, sample_weight = np.array(y * (sample_weight_coef) + 1))
        grid_search_results = pd.DataFrame(grid_search.cv_results_)
        grid_search_results['sample_weight'] = sample_weight_coef
        cv_df_list.append(grid_search_results)

    return(pd.concat(cv_df_list, axis = 0))

train = False
test = False
study_models = True
targets = 10

if(train == True):
    error = [0, 0]

    for i in range(1, targets + 1):
        df = pd.read_csv('../inputs/train_past_3_future_10_cutoff_1_target_%d_train.csv' % (i))

        keep_columns = [x for x in df.columns.tolist() if x not in ['week', 'target']]

        df_train = df.loc[df['week'] <= max(df['week']) - 10, :]
        # df_train = df.copy()
        df_test = df.loc[df['week'] > max(df['week']) - 10, :]
        del df

        X_train = df_train[keep_columns].as_matrix()
        y_train = df_train['target'].as_matrix()
        del df_train

        X_test = df_test[keep_columns].as_matrix()
        y_test = df_test['target'].as_matrix()
        del df_test

        gc.collect()

        parameters = {'learning_rate'    : 0.1,
                      'max_depth'        : 5,
                      'min_child_weight' : 20,
                      'random_seed'      : 42,
                      'n_estimators'     : 50,
                      'subsample'        : 0.8,
                      'colsample_bytree' : 0.8}

        search_params_dict = {
            'learning_rate'    : [0.1],
            'max_depth'        : [3, 5],
            'min_child_weight' : [20, 50],
            'random_seed'      : [42],
            'n_estimators'     : [50],
            'subsample'        : [0.8, 1.0],
            'colsample_bytree' : [0.3, 0.8]
        }

        model = LGBMRegressor(**parameters)

        model.fit(X_train, y_train)
        # grid_search = grid_search_sample_weight(model, X_train, y_train, search_params_dict)
        # grid_search.to_csv('../models/model_%d_grid_search.csv' % (i), index = False)        
        pred = model.predict(X_test)

        with open('../models/model_%d' % (i), 'wb') as f:
            pickle.dump(model, f)

        print(i, ' rmse : ', rmse(y_test, pred))

        error[0] += se(y_test, pred)
        error[1] += y_test.shape[0]

    print('total rmse ', np.sqrt(error[0]/error[1]))

if(test == True):
    df_list = []
    for i in range(1, targets + 1):
        df = pd.read_csv('../inputs/train_past_3_future_10_cutoff_1_target_%d_test.csv' % (i))
        df = df.loc[df['week'] == 145]

        keep_columns = [x for x in df.columns.tolist() if x not in ['week', 'target']]

        with open('../models/model_%d' % (i), 'rb') as f:
            model = pickle.load(f)

        df['num_orders'] = model.predict(df[keep_columns])
        df = df[['week', 'center_id', 'meal_id', 'num_orders']].copy()
        df['week'] = df['week'].apply(lambda x: i + x)
        df_list.append(df.copy())

    df = pd.concat(df_list)
    df_test = pd.read_csv('../inputs/test.csv')
    df = pd.merge(left = df_test, right = df, on = ['week', 'center_id', 'meal_id'], how = 'left')

    print('Null ids ', (df['id'].isnull()).sum())
    df = df.loc[~df['id'].isnull(), :]
    print('Shape ', df.shape)

    df = df[['id', 'num_orders']].copy()
    df.loc[df['num_orders'] <= 0, 'num_orders'] = 1
    df.fillna({'num_orders':df['num_orders'].mean()}, inplace = True)
    # conver to exponent
    df['num_orders'] = np.exp(df['num_orders'])

    df[['id', 'num_orders']].to_csv('../submissions/%s.csv' % (get_curr_dt()), index = False)


if(study_models == True):
    # get feature names
    df = pd.read_csv('../inputs/train_past_3_future_10_cutoff_1_target_%d_train.csv' % (1), nrows = 10)
    keep_columns = [x for x in df.columns.tolist() if x not in ['week', 'target']]

    df_list = []
    for i in range(1, targets + 1):
        with open('../models/model_%d' % (i), 'rb') as f:
            model = pickle.load(f)

        df = pd.DataFrame({'feature':keep_columns, 'importance':model.feature_importances_})
        df['model_target'] = i
        df_list.append(df.copy())

    pd.concat(df_list)[['model_target', 'feature', 'importance']].to_csv('../models/feature_importance.csv', 
                                                                         index = False)

