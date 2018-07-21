import pandas as pd
import numpy as np
import sys
import os
import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from scipy import stats
from scipy.optimize import minimize

# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import ml_utils as ml_utils


def revenue_optimize(df):
    print('Optimizing Incentives !')
    # the dataframe has columns for renewal and premium
    premium_vector = df['premium']
    renewal_vector = df['renewal']

    def revenue_func(x):
        incentive = x
        effort      = 10 * (1- np.exp(-incentive/400.0))
        delta_p_pct = 0.2 * (1 - np.exp(-effort/5.0))
        return(-1.0 * (min(1, proba * (1 + delta_p_pct)) * premium - incentive))

    incentives_vector = []
    for i in range(len(premium_vector)):
        if((i + 1) % 2000 == 0):
            print('%d Records Optimized' % (i + 1) )
        premium = premium_vector[i]
        proba   = renewal_vector[i]
        incentives_vector.append(minimize(revenue_func, 100, bounds = [(0, None)]).x[0])

    return(incentives_vector)


do_ensemble_ = False

# main procedure starts here
df = pd.read_csv('../inputs/train.csv')

fillna_dict = {'count_3-6_months_late'          : 0, 
               'count_6-12_months_late'         : 0,
               'count_more_than_12_months_late' : 0,
               'application_underwriting_score' : 90}
df, df_grouping = ml_utils.etl_pipeline(df, is_train = True, 
                                        create_interactions = False, 
                                        fillna_dict = fillna_dict,
                                        single_interaction_vars = None,
                                        higher_interaction_vars = None,
                                        target = 'renewal')


input_cols = [x for x in df.columns.tolist() \
   if x not in['renewal', 'residence_area_type', 'sourcing_channel', 'id']]

X = df[input_cols].as_matrix()
y = df['renewal'].as_matrix()

## code for grid search
parameters = {'learning_rate'    : 0.1,
              'max_depth'        : 5,
              'min_child_weight' : 100,
              'random_seed'      : 42,
              'n_estimators'     : 50,
              'subsample'        : 0.8}


model = LGBMClassifier(**parameters)
# model.fit(X, y)
model_list = ml_utils.cross_val(df, 'renewal', model, cv = 10, sample_weight_coef = 1, 
              feature_names = input_cols)

'''
df_feature_importance = pd.DataFrame(list(zip(input_cols, \
                                          model.feature_importances_)),\
                                 columns = ['column_name', 'feature_importance'])
df_feature_importance = df_feature_importance.sort_values(by = 'feature_importance', ascending = False).reset_index()
print(df_feature_importance)
    index                       column_name  feature_importance
0       3             count_3-6_months_late                 128
1       0  perc_premium_paid_by_cash_credit                 123
2       6    application_underwriting_score                  94
3       1                       age_in_days                  79
4       5    count_more_than_12_months_late                  73
5       4            count_6-12_months_late                  72
6       2                            income                  66
7       9             count_total_prem_paid                  66
8      14    total_prem_am_paid_cash_credit                  30
9       7               no_of_premiums_paid                  29
10     11                total_prem_am_paid                  14
11     13                    prem_by_income                  10
12      8                           premium                   9
13     18                sourcing_channel_A                   6
14     15         residence_area_type_Urban                   4
15     20                sourcing_channel_D                   4
16     10                          plan_age                   0
17     12                         age_years                   0
18     16         residence_area_type_Rural                   0
19     17                sourcing_channel_C                   0
20     19                sourcing_channel_B                   0
21     21                sourcing_channel_E                   0
'''

'''
# sample weight 1  works best
sample_weight_list = [1]
model = LGBMClassifier()
ml_utils.grid_search_sample_weight(model, X, y, parameters, cv = 10,\
                                   sample_weight_list = sample_weight_list)\
    .to_csv('xgboost_grid_search_cv.csv', index = False)
'''


if(True):
    # do the predictions on the test set
    df_test = pd.read_csv('../inputs/test.csv')
    df_test = ml_utils.etl_pipeline(df_test, 
                                    is_train = False, 
                                    create_interactions = False,
                                    fillna_dict = fillna_dict,
                                    single_interaction_vars = None,
                                    higher_interaction_vars = None,
                                    df_grouping_train = df_grouping,
                                    target = 'renewal')

    df_submit = pd.DataFrame(df_test[['id', 'premium']])
    
    if (do_ensemble_ == True):
        df_ensemble = pd.DataFrame(df_test[['id']])
        for model_tuple in model_list_ensemble:
            for index, model in enumerate(model_tuple[1]):
                df_ensemble[model_tuple[0] + '_' +  str(index)] = model.predict_proba(df_test[model_tuple[2]].as_matrix())[:, 1]
        
        X_test = df_ensemble[input_cols].as_matrix()
    else:
        X_test = df_test[input_cols].as_matrix()

    df_submit['renewal'] = np.mean([clf.predict_proba(X_test)[:,1] for clf in model_list], axis = 0)
    df_submit['incentives'] = revenue_optimize(df_submit)
    df_submit = df_submit[['id', 'renewal', 'incentives']]

    df_submit.to_csv('../submissions/submit_20180721_1645_lgbm_basic.csv', index=False)
