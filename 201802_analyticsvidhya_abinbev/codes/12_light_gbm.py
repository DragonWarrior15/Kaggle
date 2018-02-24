import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from scipy import stats

from lightgbm import LGBMRegressor

from input_cols_list import input_cols

def competition_metric(y_true, y_forecast):
    return(1 - (np.sum(np.abs(y_true - y_forecast)) / np.sum(y_true)))

def competition_scorer(estimator, X, y_true):
    return(1 - (np.sum(np.abs(y_true - estimator.predict(X))) / np.sum(y_true)))

def month_adder(month, diff):
    ## add integral month differences to months of the form
    ## 201701, 201712 etc
    ## input -> 201712, 10
    ## output -> 201810
    diff = int(diff)
    delta = 1 if diff > 0 else -1
    while diff != 0:
        month += delta
        diff += -1 * delta

        if 1 <= month % 100 and month % 100 <= 12:
            pass
        elif month % 100 == 13:
            month = ((month // 100) + 1) * 100 + 1
        else:
            month = ((month // 100) - 1) * 100 + 12
    return month

# get a list of all months for iteration
var_to_group = [['Agency', 'SKU'],
                ['Agency'],
                ['SKU']]
agg_list = [[np.sum],
            [np.sum, np.mean, np.max, np.min, np.std],
            [np.mean, np.max, np.min, np.std]]

def feature_prep(month):
    # start preparing the features, current month will be the target month
    # and features will be historical time window trends
    df_targets = []
    if month < 201401:
        print ('month should be greater than 201401 to prepare 12 month features')
        return (None)

    end_month = month
    
    # for month in all_month_list:
    df_temp_inputs = pd.DataFrame()
    for time in [1, 2, 3, 4, 6, 12]:
        
        agg_list_index = 0 if time == 1 else 1
        start_month = month_adder(end_month, -1 * time)

        for index, var_list in enumerate(var_to_group):
            # a second for loop is implemented especially for time = 2
            # to get the values in only that month
            for time_diff in [0, 1] if time == 2 else [0]:
                if time == 2 and time_diff == 1:
                    df_temp = df.loc[(end_month - 2 <= df['YearMonth']) & (df['YearMonth'] <= end_month - 2), :]
                else:                
                    df_temp = df.loc[(start_month <= df['YearMonth']) & (df['YearMonth'] < end_month), :]                
                df_temp = df_temp.groupby(var_list).agg({'Volume':agg_list[agg_list_index], 'Price':agg_list[agg_list_index],
                                                             'Sales':agg_list[agg_list_index]})
                if time == 2 and time_diff == 1:
                    new_col_names =  var_list + ['_'.join(['_'.join(var_list + [str(time)] + ['1'])] + list(x)) for x in (df_temp.columns.ravel())]
                else:
                    new_col_names =  var_list + ['_'.join(['_'.join(var_list + [str(time)])] + list(x)) for x in (df_temp.columns.ravel())]
                df_temp.reset_index(inplace = True)
                df_temp.columns = new_col_names

                if index == 0 and time == 1:
                    df_temp_inputs = df_temp.loc[:, :]
                else:
                    df_temp_inputs = pd.merge(left = df_temp_inputs, right = df_temp, on = var_list, how = 'left')


        # get the industry level volume features
        df_temp = df_ind.loc[(start_month <= df_ind['YearMonth']) & (df_ind['YearMonth'] < end_month), :]
        df_temp['dummy'] = 1
        df_temp = df_temp.groupby('dummy').agg({'Soda_Volume':agg_list[agg_list_index], 'Industry_Volume':agg_list[agg_list_index]})
        new_col_names = ['dummy'] + ['_'.join(['_'.join(['industry'] + [str(time)])] + list(x)) for x in (df_temp.columns.ravel())]
        df_temp.reset_index(inplace = True)
        df_temp.columns = new_col_names


        for col in df_temp.columns.tolist():
            if col != 'dummy':
                df_temp_inputs[col] =  df_temp.loc[df_temp['dummy'] == 1, col].values[0]

    
        agg_list_index = 0 if time == 1 else 2

        
        # get the Agency level weather features
        df_temp = df_weather.loc[(start_month <= df_weather['YearMonth']) & (df_weather['YearMonth'] < end_month), :]
        df_temp = df_temp.groupby(var_to_group[1]).agg({'Avg_Max_Temp':agg_list[agg_list_index]})
        new_col_names = var_to_group[1] + ['_'.join(['_'.join(var_to_group[1] + [str(time)])] + list(x)) for x in (df_temp.columns.ravel())]
        df_temp.reset_index(inplace = True)
        df_temp.columns = new_col_names
        df_temp_inputs = pd.merge(left = df_temp_inputs, right = df_temp, on = var_to_group[1], how = 'left')

    
    # add indicators from the calendar table for the target month
    for col in df_calendar.columns.tolist():
        if col != 'YearMonth':
            df_temp_inputs[col] =  df_calendar.loc[df_calendar['YearMonth'] == end_month, col].values[0]

    # add the target
    df_temp = df.loc[df['YearMonth'] == end_month, ['Agency', 'SKU', 'Volume']]
    df_temp = df_temp.rename(columns = {'Volume':'target'})
    df_temp_inputs = pd.merge(left = df_temp_inputs, right = df_temp, on = ['Agency', 'SKU'], how = 'left')

    # remove the rows where the target is zero
    if month == 201801:
        pass
    else:
        df_temp_inputs = df_temp_inputs.loc[df_temp_inputs['target'] > 0, :]

    return (df_temp_inputs)



## read the inputs
df_price       = pd.read_csv('../inputs/price_sales_promotion.csv')
df_demo        = pd.read_csv('../inputs/demographics.csv')
df_ind_sales   = pd.read_csv('../inputs/industry_soda_sales.csv')
df_ind_volume  = pd.read_csv('../inputs/industry_volume.csv')
df_hist_volume = pd.read_csv('../inputs/historical_volume.csv')
df_calendar    = pd.read_csv('../inputs/event_calendar.csv')
df_weather     = pd.read_csv('../inputs/weather.csv')

df_demo['Total_Income_2017'] = df_demo['Avg_Population_2017'] * df_demo['Avg_Yearly_Household_Income_2017']
df_demo['Avg_Income_2017'] = df_demo['Total_Income_2017'] / np.sum(df_demo['Total_Income_2017'])
df_demo['Avg_Population_2017'] = df_demo['Avg_Population_2017'] / np.sum(df_demo['Avg_Population_2017'])
df_demo = df_demo[['Agency', 'Avg_Population_2017', 'Avg_Income_2017']]

df_calendar = df_calendar.drop(['Regional Games ', 'FIFA U-17 World Cup', 'Football Gold Cup'], axis = 1)

## join the relevant inputs
df = pd.merge(left = df_price, right = df_hist_volume, on = ['Agency', 'SKU', 'YearMonth'], how = 'inner')

# df = pd.merge(left = df, right = df_demo, on = ['Agency'])
# df = pd.merge(left = df, right = df_weather, on = ['Agency', 'YearMonth'], how = 'inner')
# df = pd.merge(left = df, right = df_calendar, on = ['YearMonth'])

df_ind = pd.merge(left = df_ind_sales, right = df_ind_volume, on = ['YearMonth'])

# print (df.columns.values)

# start preparing the input table

all_month_list = sorted(df['YearMonth'].unique().tolist())
df_input = []
for month in all_month_list:
    if 201401 <= month and month <= 201712:
        print (month)
        df_input.append(feature_prep(month).loc[:, :])

df_input = pd.concat(df_input)
print (len(df_input))
print (df_input.columns)


df_input = df_input.drop(['Agency', 'SKU'], axis = 1)
# X = df_input[[x for x in df_input.columns.tolist() if x != 'target']].as_matrix()
X = df_input[input_cols].as_matrix()
y = df_input['target'].as_matrix()

'''
## code for grid search
parameters = {'learning_rate':[0.01, 0.05, 0.1],
              'n_estimators':[10, 25, 50, 100],
              'max_depth':[2, 3, 5, 8],
              'subsample':[0.8, 0.9],
              'min_child_weight':[1, 2, 5],
              'random_seed':[42]}

model = LGBMRegressor()
grid_search = GridSearchCV(model, parameters, scoring = competition_scorer, cv = 3)

grid_search.fit(X, y)

grid_search_results = pd.DataFrame(grid_search.cv_results_)
grid_search_results.to_csv('temp.csv', index = False)
'''


model_list = []
kf = KFold(n_splits = 4, shuffle = True)
kf_index = 0
for train_indices, test_indices in kf.split(X):
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    ## best parameters from the CV
    model = LGBMRegressor(learning_rate = 0.1, max_depth = 8, min_child_weight = 1, n_estimators = 100, random_seed = 42, subsample = 0.9)
        
    model.fit(X_train, y_train)
    model_list.append(model)
    kf_index += 1
    
    print(str(kf_index) + ',' + ' Train : ' + str(round(competition_metric(y_train, model.predict(X_train)), 5)) + \
          ' , Test : ' + str(round(competition_metric(y_test, model.predict(X_test)), 5)))
    # print the errors for Jan months
    # print(str(kf_index) + ',' + ' Train : ' + str(round(competition_metric(y_train, model.predict(X_train)), 5)) + \
          # ' , Test : ' + str(round(competition_metric(y_test, model.predict(X_test)), 5)))

df_feature_importance = pd.DataFrame(list(zip([x for x in input_cols if x != 'target'], \
                                              model.feature_importances_)),\
                                     columns = ['column_name', 'feature_importance'])
df_feature_importance.to_csv('temp2.csv', index = False)
