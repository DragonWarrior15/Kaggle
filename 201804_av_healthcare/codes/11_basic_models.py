import pandas as pd
import numpy as np
import sys
import os
import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from scipy import stats

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

def cross_val(X, y, model, cv = 3):
    model_list = []
    kf = KFold(n_splits = cv, shuffle = True)
    kf_index = 0
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        print (np.sum(y_train), np.sum(y_test))

        sample_weight = np.array(y_train * (1/0.018) + 1)

        model.fit(X_train, y_train, sample_weight = sample_weight)
        model_list.append(model)
        kf_index += 1

        print (np.sum(model.predict(X_train)), np.sum(model.predict(X_test)))
        
        print(str(kf_index) + ',' + ' Train : ' + str(format(roc_auc_score(y_train, model.predict(X_train)), '.5f')) + \
              ' , Test : ' + str(format(roc_auc_score(y_test, model.predict(X_test)), '.5f')))
    
    return (model_list)

def age_buckets_fun(age):
    if age < 20:
        return ('0.[0,20)')
    if age < 45:
        return ('1.[20,45)')
    if age < 54:
        return ('2.[45,54)')
    if age < 65:
        return ('3.[54,65)')
    if age < 74:
        return ('4.[65,74)')
    if age < 84:
        return ('5.[75,84)')
    return ('6.[85,)')

def glucose_bucket_fun(glucose):
    if glucose < 80:
        return ('0.[0,80)')
    if glucose < 100:
        return ('1.[80,100)')
    if glucose < 200:
        return ('2.[100,200)')
    return ('3.[200,)')

def bmi_bucket_fun(bmi):
    if bmi < 0:
        return ('0.NA')
    if bmi < 18.5:
        return ('1.[,18.5)')
    if bmi < 24.9:
        return ('2.[18.5,24.9)')
    if bmi < 29.9:
        return ('3.[24.9,29.9)')
    return ('4.[29.9,)')

df = pd.read_csv('../inputs/train.csv')

# create buckets for continuous variables
df['age_buckets'] = df['age'].apply(age_buckets_fun)
df['glucose_bucket'] = df['avg_glucose_level'].apply(glucose_bucket_fun)
df['bmi_bucket'] = df['bmi'].apply(bmi_bucket_fun)

# null handling
df.loc[df['gender'] == 'Other', 'gender'] = 'Male'
df.loc[df['work_type'] == 'Never_worked', 'work_type'] = 'children'
df['bmi'].fillna(-1, inplace = True)
df['smoking_status'].fillna('NA', inplace = True)

# create one hot encoded variables
vars_to_one_hot_encode = ['gender', 'hypertension', 'heart_disease',
                          'ever_married', 'work_type', 'Residence_type',
                          'smoking_status', 'age_buckets', 'glucose_bucket',
                          'bmi_bucket' ]

vars_for_model = []

for col in vars_to_one_hot_encode:
    for val in df[col].unique():
        vars_for_model.append(col + '_' + str(val))
        # print (vars_for_model[-1])
        df[col + '_' + str(val)] = df[col].apply(lambda x: 1 if x == val else 0)

X = df[vars_for_model].as_matrix()
y = df['stroke'].as_matrix()
# print (X.shape, y.shape)
# print (np.sum(X), np.sum(y))
# for i in range(X.shape[1]):
    # print (vars_for_model[i], i, np.sum(X[:,i]))

'''
params for lightgbm
param_learning_rate    0.1
param_max_depth        2
param_min_child_weight 5
param_n_estimators     50
param_random_seed      42
param_subsample        0.8
sample_weight_coef     55
'''

'''
## code for grid search
cv_df_list = []
for index, sample_weight_coef in enumerate([55, 20, 10, 5]):
    parameters = {'learning_rate':[0.01, 0.05, 0.1],
                  'n_estimators':[10, 25, 50, 100],
                  'max_depth':[2, 3, 5, 8],
                  'subsample':[0.8, 0.9],
                  'min_child_weight':[1, 2, 5],
                  'random_seed':[42]
                  }

    model = LGBMClassifier()
    grid_search = GridSearchCV(model, parameters, scoring = 'roc_auc', cv = 10)
    grid_search.fit(X, y, sample_weight = np.array(y * (sample_weight_coef) + 1))
    grid_search_results = pd.DataFrame(grid_search.cv_results_)
    grid_search_results['sample_weight'] = sample_weight_coef
    cv_df_list.append(grid_search_results)

pd.concat(cv_df_list).to_csv('temp.csv', index = False)
'''


## code for grid search
cv_df_list = []
for index, sample_weight_coef in enumerate([55, 20]):
    parameters = {'learning_rate':[0.01, 0.1],
                  'n_estimators':[10, 25, 50],
                  'max_depth':[2, 3, 5],
                  'subsample':[0.8, 0.9],
                  'min_child_weight':[1, 2],
                  'colsample_bytree':[0.6, 0.8],
                  'random_seed':[42]
                  }

    model = XGBClassifier()
    grid_search = GridSearchCV(model, parameters, scoring = 'roc_auc', cv = 10)
    grid_search.fit(X, y, sample_weight = np.array(y * (sample_weight_coef) + 1))
    grid_search_results = pd.DataFrame(grid_search.cv_results_)
    grid_search_results['sample_weight'] = sample_weight_coef
    cv_df_list.append(grid_search_results)

pd.concat(cv_df_list).to_csv('temp.csv', index = False)


'''
## code for grid search
cv_df_list = []
for index, sample_weight_coef in enumerate([55, 20, 10, 5]):
    parameters = {'learning_rate':[1, 0.1],
                  'iterations':[10, 25, 50],
                  'depth':[2, 3, 5, 8],
                  'random_seed':[42]
                  }
    print ('Current sample weight : ' + str(sample_weight_coef) + ', started at ' + str(time.asctime()))
    start_time = time.time()

    model = CatBoostClassifier()
    grid_search = GridSearchCV(model, parameters, scoring = 'roc_auc', cv = 10)
    grid_search.fit(X, y, sample_weight = np.array(y * (sample_weight_coef) + 1))
    grid_search_results = pd.DataFrame(grid_search.cv_results_)
    grid_search_results['sample_weight'] = sample_weight_coef
    cv_df_list.append(grid_search_results)

    end_time = time.time()
    print ('Iteration done in ' + str(int(end_time - start_time)) + 's at ' + str(time.asctime()))

pd.concat(cv_df_list).to_csv('temp.csv', index = False)

# model_list = cross_val(X, y, model, cv = 10)
'''