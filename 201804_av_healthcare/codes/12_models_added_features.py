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

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def auc_keras(y_true, y_pred):
    # y_true = K.eval(y_true)
    # y_pred = K.eval(y_pred)
    return K.variable(1. - roc_auc_score(y_true,y_pred))

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
    df.loc[df['gender'] == 'Other', 'gender'] = 'Male'
    df.loc[df['work_type'] == 'Never_worked', 'work_type'] = 'children'

    df_etl['age_bucket'] = df_etl['age'].apply(age_buckets_fun)
    df_etl['glucose_bucket'] = df_etl['avg_glucose_level'].apply(glucose_bucket_fun)
    df_etl['bmi_bucket'] = df_etl['bmi'].apply(bmi_bucket_fun)

    vars_to_one_hot_encode = ['gender', 'hypertension', 'heart_disease',
                          'ever_married', 'work_type', 'residence_type',
                          'smoking_status', 'age_bucket', 'glucose_bucket',
                          'bmi_bucket']

    # encode basic variables
    for col in vars_to_one_hot_encode:
        for val in df_etl[col].unique():
            # vars_for_model.append(col + '_' + str(val))
            # print (vars_for_model[-1])
            df_etl[col + '_' + str(val)] = df_etl[col].apply(lambda x: 1 if x == val else 0)
    
    # fill missing values
    if fillna_dict is not None:
        df_etl.fillna(fillna_dict, inplace = True)
    
    if create_interactions == True and is_train == True:
        if single_interaction_vars is not None:
            df_grouping = {}
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
        if (len(df_grouping) > 0):
            return (df_etl, df_grouping)
    return (df_etl)

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

df = pd.read_csv('../inputs/train.csv')

single_interaction_vars = ['gender', 'hypertension', 'heart_disease',
                           'ever_married', 'work_type', 'residence_type',
                           'smoking_status', 'age_bucket', 'glucose_bucket',
                           'bmi_bucket']
higher_interaction_vars = [('age_bucket', 'bmi_bucket'),
                           ('work_type', 'smoking_status'),
                           ('age_bucket', 'glucose_bucket'),
                           ('hypertension', 'heart_disease'),
                           ('hypertension', 'age_bucket'),
                           ('heart_disease', 'smoking_status'),
                           ('hypertension', 'smoking_status'),
                           ('glucose_bucket', 'heart_disease'),
                           ('heart_disease', 'gender'),
                           ('bmi_bucket', 'heart_disease'),
                           ('age_bucket', 'heart_disease'),
                           ('smoking_status', 'heart_disease', 'age_bucket'),
                           ('glucose_bucket', 'heart_disease', 'age_bucket'),
                           ('glucose_bucket', 'heart_disease', 'hypertension'),
                           ('smoking_status', 'heart_disease', 'hypertension')]

feature_importances_dict_lgbm = {
                                                  'age':    35,
                                    'avg_glucose_level':    10,
                                                  'bmi':    46,
                                          'gender_Male':     0,
                                        'gender_Female':     0,
                                       'hypertension_0':     0,
                                       'hypertension_1':     0,
                                      'heart_disease_0':     0,
                                      'heart_disease_1':     0,
                                      'ever_married_No':     0,
                                     'ever_married_Yes':     0,
                                   'work_type_children':     0,
                                    'work_type_Private':     0,
                              'work_type_Self-employed':     0,
                                   'work_type_Govt_job':     0,
                                 'residence_type_Rural':     0,
                                 'residence_type_Urban':     0,
                                   'smoking_status_nan':     0,
                          'smoking_status_never smoked':     0,
                       'smoking_status_formerly smoked':     0,
                                'smoking_status_smokes':     1,
                                  'age_bucket_0.[0,20)':     0,
                                 'age_bucket_3.[54,65)':     0,
                                 'age_bucket_4.[65,74)':     0,
                                 'age_bucket_2.[45,54)':     0,
                                 'age_bucket_5.[75,84)':     0,
                                 'age_bucket_1.[20,45)':     0,
                            'glucose_bucket_1.[80,100)':     0,
                           'glucose_bucket_2.[100,200)':     0,
                              'glucose_bucket_0.[0,80)':     0,
                              'glucose_bucket_3.[200,)':     0,
                                 'bmi_bucket_1.[,18.5)':     0,
                                 'bmi_bucket_4.[29.9,)':     0,
                             'bmi_bucket_2.[18.5,24.9)':     0,
                             'bmi_bucket_3.[24.9,29.9)':     0,
                                   'gender_stroke_mean':     0,
                             'hypertension_stroke_mean':     0,
                            'heart_disease_stroke_mean':     0,
                             'ever_married_stroke_mean':     0,
                                'work_type_stroke_mean':     0,
                           'residence_type_stroke_mean':     0,
                           'smoking_status_stroke_mean':     0,
                               'age_bucket_stroke_mean':     0,
                           'glucose_bucket_stroke_mean':     0,
                               'bmi_bucket_stroke_mean':     0,
                    'age_bucket_bmi_bucket_stroke_mean':     4,
                 'work_type_smoking_status_stroke_mean':     5,
                'age_bucket_glucose_bucket_stroke_mean':     1,
               'hypertension_heart_disease_stroke_mean':     0,
                  'hypertension_age_bucket_stroke_mean':     3,
             'heart_disease_smoking_status_stroke_mean':     0,
              'hypertension_smoking_status_stroke_mean':     1,
             'glucose_bucket_heart_disease_stroke_mean':     0,
                     'heart_disease_gender_stroke_mean':     1,
                 'bmi_bucket_heart_disease_stroke_mean':     0,
                 'age_bucket_heart_disease_stroke_mean':     0,
  'smoking_status_heart_disease_age_bucket_stroke_mean':    23,
  'glucose_bucket_heart_disease_age_bucket_stroke_mean':    13,
'glucose_bucket_heart_disease_hypertension_stroke_mean':     3,
'smoking_status_heart_disease_hypertension_stroke_mean':     4
}

df, df_grouping = etl_pipeline(df, is_train = True, 
                               create_interactions = True, 
                               fillna_dict = {'bmi':-1, 
                                              'smoking_status':'NA'},
                               single_interaction_vars = single_interaction_vars,
                               higher_interaction_vars = higher_interaction_vars,
                               target = 'stroke')

do_ensemble_ = True
model_list_ensemble = []

# model = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 20)

# logistic regression
print ('\nTraining Logistic Regression')
params = {'penalty'      : ['l1'],
          'C'            : [0.1],
          'random_state' : [42, 1000]}

for params in get_param_space(params):
    print (params)
    model = LogisticRegression(random_state   = params['random_state'],
                               fit_intercept  = True)

    vars_for_model = [x for x in df.columns.tolist() if x not in \
                                list(['stroke', 'id'] + single_interaction_vars )]

    model_list = cross_val(df, 'stroke', model, cv = 10, sample_weight_coef = 55, 
                           feature_names = vars_for_model,
                           eval_score = roc_auc_score,
                           eval_metric = 'roc_auc')

    model_list_ensemble.append(['lr', model_list, vars_for_model])

# lightgbm
print ('Training LightGBM')
params = {'learning_rate'    : [0.1],
          'max_depth'        : [2],
          'min_child_weight' : [1],
          'random_seed'      : [42, 1000],
          'n_estimators'     : [50],
          'subsample'        : [0.8]}

for params in get_param_space(params):
    print (params)
    model = LGBMClassifier(learning_rate    = params['learning_rate'],
                           max_depth        = params['max_depth'],
                           min_child_weight = params['min_child_weight'],
                           n_estimators     = params['n_estimators'],
                           subsample        = params['subsample'],
                           random_seed      = params['random_seed'])

    vars_for_model = [x for x in df.columns.tolist() if x not in \
                                list(['stroke', 'id'] + single_interaction_vars )
                                and feature_importances_dict_lgbm[x] > 0]

    model_list = cross_val(df, 'stroke', model, cv = 10, sample_weight_coef = 55, 
                           feature_names = vars_for_model,
                           eval_score = roc_auc_score,
                           eval_metric = 'roc_auc')

    model_list_ensemble.append(['lgbm', model_list, vars_for_model])


# random forest
print ('\nTraining Random Forest')
params = {'max_depth'        : [2],
          'n_estimators'     : [50],
          'random_state'      : [42, 1000]}

for params in get_param_space(params):
    print (params)
    model = RandomForestClassifier(max_depth    = params['max_depth'],
                                   n_estimators = params['n_estimators'],
                                   random_state = params['random_state'])

    vars_for_model = [x for x in df.columns.tolist() if x not in \
                                list(['stroke', 'id'] + single_interaction_vars )]

    model_list = cross_val(df, 'stroke', model, cv = 10, sample_weight_coef = 55, 
                           feature_names = vars_for_model,
                           eval_score = roc_auc_score,
                           eval_metric = 'roc_auc')

    model_list_ensemble.append(['rf', model_list, vars_for_model])


# ensemble
print ('\nTraining Ensemble')
params = {'penalty'      : 'l1',
          'C'            : 0.1,
          'random_state' : 42}

# form the new df with outputs of the models as features
df_ensemble = pd.DataFrame(df[['id', 'stroke']])
for model_tuple in model_list_ensemble:
    for index, model in enumerate(model_tuple[1]):
        df_ensemble[model_tuple[0] + '_' +  str(index)] = model.predict_proba(df[model_tuple[2]].as_matrix())[:, 1]

vars_for_model = [x for x in df_ensemble.columns.tolist() if x not in ['id', 'stroke']]

model = LogisticRegression(random_state   = params['random_state'],
                           fit_intercept  = True)
    
model_list = cross_val(df_ensemble, 'stroke', model, cv = 10, sample_weight_coef = 55, 
                       feature_names = vars_for_model,
                       eval_score = roc_auc_score,
                       eval_metric = 'roc_auc')

'''
# catboost
params = {'learning_rate' : 0.1,
          'depth'         : 2,
          'iterations'    : 50,
          'random_seed'   : 42}

vars_for_model = [x for x in df.columns.tolist() if x not in \
                            list(['stroke', 'id'] + single_interaction_vars )]
                            # and feature_importances_dict[x] > 0]

model = CatBoostClassifier(learning_rate = params['learning_rate'],
                           depth         = params['depth'],
                           iterations    = params['iterations'],
                           random_seed   = params['random_seed'])

'''

'''
# NN
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K

vars_for_model = [x for x in df.columns.tolist() if x not in \
                            list(['stroke', 'id'] + single_interaction_vars )]

model = Sequential()
model.add(BatchNormalization(input_shape = (len(vars_for_model),)))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 10, activation = 'relu'))
model.add(Dense(units = 5, activation = 'relu'))
model.add(Dense(units = 1, activation='sigmoid'))

adam = Adam(lr=0.0001)
sgd = SGD(lr=1)
model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])

model_list = cross_val(df, 'stroke', model, cv = 10, sample_weight_coef = 0, 
                       feature_names = vars_for_model,
                       eval_score = roc_auc_score,
                       eval_metric = 'roc_auc',
                       model_type = 'NN')
'''

'''
## code for grid search
cv_df_list = []
for index, sample_weight_coef in enumerate([55, 20]):
    parameters = {'learning_rate':[0.1, 0.01],
                  'n_estimators':[30, 50, 80],
                  'max_depth':[2, 3, 5],
                  'subsample':[0.8],
                  'min_child_weight':[1, 2],
                  'random_seed':[42]
                  }

    print ('Current sample weight : ' + str(sample_weight_coef) + ', started at ' + str(time.asctime()))
    start_time = time.time()

    model = LGBMClassifier()
    X = df[vars_for_model].as_matrix()
    y = df['stroke'].as_matrix()
    grid_search = GridSearchCV(model, parameters, scoring = 'roc_auc', cv = 10)
    grid_search.fit(X, y, sample_weight = np.array(y * (sample_weight_coef) + 1))
    grid_search_results = pd.DataFrame(grid_search.cv_results_)
    grid_search_results['sample_weight'] = sample_weight_coef
    cv_df_list.append(grid_search_results)
    
    end_time = time.time()
    print ('Iteration done in ' + str(int(end_time - start_time)) + 's at ' + str(time.asctime()))

pd.concat(cv_df_list).to_csv('temp.csv', index = False)
'''


if(True):
    # do the predictions on the test set
    df_test = pd.read_csv('../inputs/test.csv')
    df_test = etl_pipeline(df_test, 
                           is_train = False, 
                           create_interactions = True,
                           fillna_dict = {'bmi':-1, 
                                          'smoking_status':'NA'},
                           single_interaction_vars = single_interaction_vars,
                           higher_interaction_vars = higher_interaction_vars,
                           df_grouping_train = df_grouping,
                           target = 'stroke')

    df_submit = pd.DataFrame(df_test['id'])
    
    if (do_ensemble_ == True):
        df_ensemble = pd.DataFrame(df_test[['id']])
        for model_tuple in model_list_ensemble:
            for index, model in enumerate(model_tuple[1]):
                df_ensemble[model_tuple[0] + '_' +  str(index)] = model.predict_proba(df_test[model_tuple[2]].as_matrix())[:, 1]
        
        X_test = df_ensemble[vars_for_model].as_matrix()
    else:
        X_test = df_test[vars_for_model].as_matrix()

    df_submit['stroke'] = np.mean([clf.predict_proba(X_test)[:,1] for clf in model_list], axis = 0)
    # df_submit['stroke'] = np.mean([clf.predict(X_test) for clf in model_list], axis = 0)

    df_submit.to_csv('../submissions/submit_20180415_0253_ensemble_interactions.csv', index=False)
