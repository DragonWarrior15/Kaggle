import pandas as pd
import numpy as np
import sys
import os
import time

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import export_graphviz

from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold


def cross_val(df, target, model, cv = 3, sample_weight_coef = 1, 
              feature_names = None, eval_score = roc_auc_score, eval_metric = 'roc_auc',
              model_type = None, model_fit_params = {'epochs':10, 'batch_size':512}):
    model_list = []
    kf = KFold(n_splits = cv, shuffle = True, random_state = 42)
    kf_index = 0

    X = df[feature_names].as_matrix()
    y = df[target].as_matrix()

    for train_indices, test_indices in kf.split(X, y):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        sample_weight = np.array(y_train * sample_weight_coef + 1)

        if(model_type == 'NN'):
            model.fit(X_train, y_train, 
                      epochs = model_fit_params['epochs'],
                      batch_size = model_fit_params['batch_size'],
                      sample_weight = sample_weight)
        else:
            model.fit(X_train, y_train, sample_weight = sample_weight)
            
        model_list.append(model)
        kf_index += 1

        if(eval_metric in ['roc_auc'] and model_type not in ['NN']):
            print((('%2d') % (kf_index)) + ',' + ' Train : ' + str(format(eval_score(y_train, model.predict_proba(X_train)[:, 1]), '.5f')) + \
                  ' , Test : ' + str(format(eval_score(y_test, model.predict_proba(X_test)[:, 1]), '.5f')))
        else:
            print((('%2d') % (kf_index)) + ',' + ' Train : ' + str(format(eval_score(y_train, model.predict(X_train)), '.5f')) + \
                  ' , Test : ' + str(format(eval_score(y_test, model.predict(X_test)), '.5f')))


        if(False):
            feature_importances_list = [[feature_names[i], model.feature_importances_[i]] for i in range(len(feature_names))]
            max_feature_name = max([len(x) for x in feature_names])
            print ('feature_importances')
            for i in range(len(feature_names)):
                print (('%' + str(max_feature_name) + 's: %5d') % (feature_importances_list[i][0], feature_importances_list[i][1]))

        if(False):
            export_graphviz(model, out_file = 'tree_images/tree_' + str(kf_index) + '.dot', feature_names = feature_names)
            os.system('dot -Tpng tree_images/tree_' + str(kf_index) + '.dot -o tree_images/tree_' + str(kf_index) + '.png')
    
    return (model_list)

def etl_pipeline(df, is_train = False, create_interactions = False, fillna_dict = None,\
                 single_interaction_vars = None, higher_interaction_vars = None, df_grouping_train = None,\
                 target = None):
    assert target != None
    columns = [x.lower() for x in df.columns.tolist()]
    df.columns = columns
    df_etl = df.loc[:,:]
    if is_train == True:
        # apply some filter here
        # df_etl = df_etl.loc[(~df_etl['dob'].isnull()) & (~df_etl['loan_period'].isnull())]
        # df_etl = df_etl.loc[(~df_etl['dob'].isnull())]
        pass
        
    # create new variables
    df_etl['count_total_prem_paid'] = df_etl['count_3-6_months_late'] + df_etl['count_6-12_months_late'] + \
                                      df_etl['count_more_than_12_months_late'] + df_etl['no_of_premiums_paid']
    df_etl['plan_age'] = df_etl['count_total_prem_paid'] / 12.0
    df_etl['total_prem_am_paid'] = df_etl['count_total_prem_paid'] * df['premium']
    df['age_years'] = df['age_in_days']/365.0
    df_etl['prem_by_income'] = (1.0 * df_etl['premium'])/df_etl['income']
    df_etl['total_prem_am_paid_cash_credit'] = df_etl['perc_premium_paid_by_cash_credit'] * df_etl['total_prem_am_paid']


    vars_to_one_hot_encode = ['residence_area_type', 'sourcing_channel']

    # encode basic variables
    for col in vars_to_one_hot_encode:
        for val in df_etl[col].unique():
            # vars_for_model.append(col + '_' + str(val))
            # print (vars_for_model[-1])
            df_etl[col + '_' + str(val)] = df_etl[col].apply(lambda x: 1 if x == val else 0)
    
    # fill missing values
    if fillna_dict is not None:
        df_etl.fillna(fillna_dict, inplace = True)
    
    df_grouping = {}
    if create_interactions == True and is_train == True:
        if single_interaction_vars is not None:
            for var_to_group in single_interaction_vars:
                print (var_to_group)
                df_grouping[var_to_group] = df_etl.groupby([var_to_group])\
                                                  .agg({target:[np.mean]})
                new_col_names =  [df_grouping[var_to_group].index.name] + ['_'.join([df_grouping[var_to_group].index.name] + list(x)) \
                                  for x in (df_grouping[var_to_group].columns.ravel())]
                # print (new_col_names)
                df_grouping[var_to_group].reset_index(inplace = True)
                df_grouping[var_to_group].columns = new_col_names
                for col in df_grouping[var_to_group].columns:
                    if '_count' in col:
                        df_grouping[var_to_group][col] = df_grouping[var_to_group][col]/len(df)

                df_etl = pd.merge(left = df_etl, right = df_grouping[var_to_group], on = var_to_group, how = 'left')

        if higher_interaction_vars is not None:
            for var_to_group in higher_interaction_vars:
                print (var_to_group)
                df_grouping[var_to_group] = df_etl.groupby(list(var_to_group))\
                                              .agg({target:[np.mean]})
                new_col_names =  list(var_to_group) + ['_'.join(['_'.join(var_to_group)] + list(x)) \
                                  for x in (df_grouping[var_to_group].columns.ravel())]
                df_grouping[var_to_group].reset_index(inplace = True)
                df_grouping[var_to_group].columns = new_col_names
                for col in df_grouping[var_to_group].columns:
                    if '_count' in col[-6:]:
                        df_grouping[var_to_group][col] = df_grouping[var_to_group][col]/len(df)

                df_etl = pd.merge(left = df_etl, right = df_grouping[var_to_group], on = list(var_to_group), how = 'left')
            
    if is_train == False and create_interactions == True:
        assert df_grouping_train is not None
        for var_to_group in df_grouping_train.keys():
            df_etl = pd.merge(left = df_etl, right = df_grouping_train[var_to_group], on = var_to_group, how = 'left')
        
    if is_train == True:
            return (df_etl, df_grouping)
    return (df_etl)

def get_param_space(param_dict):
    param_space = []
    param_list = sorted(list([k for k in param_dict]))
    # print (param_to_int_dict)
    for param in param_list:
        curr_param_space_length = len(param_space)
        if (curr_param_space_length == 0):
            for i in range(len(param_dict[param])):
                param_space.append([param_dict[param][i]])
        else:
            for i in range(len(param_dict[param]) - 1):
                for j in range(curr_param_space_length):
                    param_space.append(list(param_space[j]) + [param_dict[param][i]])
            for i in range(curr_param_space_length):
                param_space[i].append(param_dict[param][-1])
    # print (param_space)
    param_space2 = [dict([[param_list[j], param_space[i][j]] for j in range(len(param_list))]) for i in range(len(param_space))]
    return (param_space2)

def grid_search_sample_weight(model, X, y, \
                              search_params_dict, \
                              scoring = 'roc_auc', cv = 3, \
                              sample_weight_list = [1]):
    # code for grid search
    cv_df_list = []
    for sample_weight_coef in sample_weight_list:
        grid_search = GridSearchCV(model, search_params_dict, scoring = 'roc_auc', cv = 10)
        grid_search.fit(X, y, sample_weight = np.array(y * (sample_weight_coef) + 1))
        grid_search_results = pd.DataFrame(grid_search.cv_results_)
        grid_search_results['sample_weight'] = sample_weight_coef
        cv_df_list.append(grid_search_results)

    return(pd.concat(cv_df_list, axis = 0))
