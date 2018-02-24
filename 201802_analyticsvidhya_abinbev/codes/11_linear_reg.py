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

def competition_metric(y_true, y_forecast):
    return(1 - np.sum(np.abs(y_true, y_forecast) / y_true))

## read the inputs
df_price       = pd.read_csv('../inputs/price_sales_promotion.csv')
df_demo        = pd.read_csv('../inputs/demographics.csv')
df_ind_sales   = pd.read_csv('../inputs/industry_soda_sales.csv')
df_ind_volume  = pd.read_csv('../inputs/industry_volume.csv')
df_hist_volume = pd.read_csv('../inputs/historical_volume.csv')
df_calendar    = pd.read_csv('../inputs/event_calendar.csv')
df_weather     = pd.read_csv('../inputs/weather.csv')

## join the relevant inputs
df = pd.merge(left = df_price, right = df_hist_volume, on = ['Agency', 'SKU', 'YearMonth'], how = 'inner')

## as a first step, try to predict the volume for the month 201501, 201601 and 201701 using the historical data
df_target = df.loc[df['YearMonth'].isin([201401, 201501, 201601, 201701]), ['Agency', 'SKU', 'YearMonth', 'Volume']].sort_values(by = ['Agency', 'SKU', 'YearMonth'])

## prepare the input features at Agency X SKU X time
# separate the dataframe into three inputs, for target years 2015, 2016 and 2017
df_input = []
var_to_group = ['Agency', 'SKU']
agg_list = [np.mean, np.sum, np.var, np.max, np.min]
for i in [4, 5, 6, 7]:
    df_temp_input = []
    end_month = 201001 + i * 100
    for time in [1, 2, 3, 6, 12]:

        if time == 12:
            start_month = 201001 + (i - 1) * 100
        elif time == 24:
            start_month = 201001 + (i - 2) * 100
        else:
            start_month = 201000 + (i - 1) * 100 + (12 - time - 1)

        df_temp = df.loc[((start_month) <= df['YearMonth']) & (df['YearMonth'] < end_month), :]

        df_temp = df_temp.groupby(var_to_group).agg({'Volume':agg_list, 'Price':agg_list,
                                                     'Sales':agg_list})
        new_col_names =  var_to_group + ['_'.join(['_'.join(var_to_group + [str(time)])] + list(x)) for x in (df_temp.columns.ravel())]
        df_temp.reset_index(inplace = True)
        df_temp.columns = new_col_names

        df_temp_input.append(df_temp.loc[:,:])

    df_temp = df_temp_input[0].loc[:, :]
    for j in range(1, len(df_temp_input)):
        df_temp = pd.merge(left = df_temp, right = df_temp_input[j], on = var_to_group, how = 'inner')
    df_input.append(df_temp.loc[:, :])
    df_input[-1]['YearMonth'] = end_month

cols_to_use = ['Agency_SKU_1_Volume_mean',
               'Agency_SKU_1_Volume_sum',
               'Agency_SKU_2_Volume_mean',
               'Agency_SKU_2_Volume_sum',
               'Agency_SKU_3_Volume_mean',
               'Agency_SKU_3_Volume_sum',
               'Agency_SKU_1_Volume_amin',
               'Agency_SKU_12_Volume_mean',
               'Agency_SKU_12_Volume_sum',
               'Agency_SKU_6_Volume_mean',
               'Agency_SKU_6_Volume_sum',
               'Agency_SKU_1_Volume_amax',
               'Agency_SKU_3_Volume_amax',
               'Agency_SKU_2_Volume_amax',
               'Agency_SKU_2_Volume_amin',
               'Agency_SKU_12_Volume_amin',
               'Agency_SKU_3_Volume_amin',
               'Agency_SKU_6_Volume_amax',
               'Agency_SKU_6_Volume_amin',
               'Agency_SKU_12_Volume_amax',
               'YearMonth']

X = pd.concat(df_input).sort_values(by = ['Agency', 'SKU', 'YearMonth'])[cols_to_use]
# X.drop(['Agency', 'SKU'], axis = 1, inplace = True)

X_train = X.loc[X['YearMonth'].isin([201401, 201501, 201601]), :]
X_test  = X.loc[X['YearMonth'].isin([201701]), :]

X_train = X_train.drop('YearMonth', axis = 1).as_matrix()
X_test  = X_test.drop('YearMonth', axis = 1).as_matrix()

y_train = df_target.loc[df_target['YearMonth'].isin([201401, 201501, 201601]), 'Volume'].as_matrix()
y_test  = df_target.loc[df_target['YearMonth'].isin([201701]), 'Volume'].as_matrix()

## standard scale the values
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

## train the model
model = LinearRegression(fit_intercept = True, normalize = False, n_jobs = -1)
model.fit(X_train, y_train)

print (X.columns)
for i in range(X_train.shape[1]):
    print (X.columns.tolist()[i], np.corrcoef(X_train[:, i], y_train)[0][1])
print (model.coef_)
print (model.intercept_)

y_train_pred = model.predict(X_train)
print (competition_metric(y_train_pred, y_train))

y_test_pred = model.predict(X_test)
print (competition_metric(y_test_pred, y_test))

## only using the january months gives very poor performance, possible
## reasons could be not modelling seasonality and very less no of data points
## compared to the available dimensions