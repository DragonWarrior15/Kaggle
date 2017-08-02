import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# df = pd.read_csv(c_vars.train_file)
# df = pd.read_csv(c_vars.train_split_train_sample, nrows = 1000)
df = pd.read_csv(c_vars.train_split_train)
# df = df[c_vars.header_useful]

df.fillna(c_vars.fillna_dict, inplace = True)

# for col in df.columns.values:
    # print (col)
    # print (df[col].unique())
# print (df.dtypes)
df.loc[:, 'datetime_day'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
df.loc[:, 'datetime_hour'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
df = df.drop('datetime', axis=1)
df.loc[:, 'browserid'] = df['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           else c_vars.browserid_map[x])

# merchant, siteid, offerid, category
df_feature = {}
for col in ['merchant', 'siteid', 'offerid', 'category']:
# for col in ['merchant']:
    df_temp = df[[col, 'click']]
    df_temp = df_temp.groupby([col]).agg(['count', np.sum])
    df_temp.reset_index(inplace = True)
    df_temp['count'] = df_temp['click', 'count']
    df_temp['num_1'] = df_temp['click', 'sum']
    df_temp['num_0'] = df_temp['count'] - df_temp['num_1']
    df_temp = df_temp[[col, 'count', 'num_0', 'num_1']]
    df_temp.columns = df_temp.columns.get_level_values(0)
    df_temp.sort_values('count', inplace = True, axis = 0, ascending = False)
    df_temp['cumul_sum'] = np.cumsum(df_temp['count'])
    df_temp_2 = df_temp.loc[df_temp['cumul_sum'] > 0.8 * len(df), :]
    df_temp = df_temp.loc[~(df_temp['cumul_sum'] > 0.8 * len(df)), :]
    df_temp = df_temp.append({col:'LESS_FREQ', 
                                  'count':np.sum(df_temp_2['count']), 
                                  'num_0':np.sum(df_temp_2['num_0']),
                                  'num_1':np.sum(df_temp_2['num_1'])},
                             ignore_index = True)
    df_temp['click_rate'] = df_temp['num_1']/df_temp['count']
    df_temp.drop(['cumul_sum'], inplace = True, axis = 1)
    df_feature[col] = df_temp.loc[:,:]

    df = pd.merge(df, df_temp, how = 'left', on = col, suffixes = ('', ''))
    df.rename(columns = {'count':'count_'+col, 'num_0':'num_0_'+col, 
                         'num_1':'num_1_'+col, 'click_rate':'click_rate_'+col}, 
              inplace = True)
    for field in ['count', 'num_0', 'num_1', 'click_rate']:
        df[field + '_' + col].fillna(df_feature[col].loc[df_temp[col] == 'LESS_FREQ', field].values[0], inplace = True)
    # print (df.columns.values)

    # print (df)
# print (df)
print (df.columns.values)

for col in ['datetime', 'click', 'merchant', 'siteid', 'offerid', 'category']:
    c_vars.header_useful.remove(col)

c_vars.header_useful.append('datetime_day')
c_vars.header_useful.append('datetime_hour')

for col in ['merchant', 'siteid', 'offerid', 'category']:
    for col2 in ['count', 'num_0', 'num_1', 'click_rate']:
        c_vars.header_useful.append(col2 + '_' + col)

print (c_vars.header_useful)
for col in ['countrycode', 'browserid', 'devid', 'datetime_day', 'datetime_hour']:
    print (df[col].unique().tolist())

X = df[c_vars.header_useful].as_matrix()
y = df['click'].as_matrix()

print (str(datetime.now()) + ' Label Encoding Started')
label_encoder = [LabelEncoder() for _ in range(5)]
for i in range(len(label_encoder)):
    label_encoder[i].fit(X[:,i])
    # print (i, c_vars.header_useful[i], label_encoder[i].get_params(deep=True))
    X[:,i] = label_encoder[i].transform(X[:,i])
print (str(datetime.now()) + ' Label Encoding Completed')

print (str(datetime.now()) + ' One Hot Encoding Started')
ohe = OneHotEncoder(sparse = False)
ohe.fit(X[:,[0,1,2,3,4]])
# print (ohe.get_params(deep=True))
X_ohe = ohe.transform(X[:,[0,1,2,3,4]])
print (str(datetime.now()) + ' One Hot Encoding Complete')
print (X.shape, X_ohe.shape)
# print (X.shape, X_ohe.shape)
# print (X, X_ohe)
X = np.hstack((X[:,[i for i in range(len(c_vars.header_useful)) if i not in [0,1,2,3,4]]], X_ohe))

# save the label encoder and the one hot encoding to disk
with open('../analysis_graphs/label_encoder', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('../analysis_graphs/one_hot_encoding', 'wb') as f:
    pickle.dump(ohe, f)
with open('../analysis_graphs/df_feature', 'wb') as f:
    pickle.dump(df_feature, f)

'''

print (X.shape, y.shape, np.sum(y))
sm = SMOTE(random_state=42)
X, y = sm.fit_sample(X, y)
print (X.shape, y.shape, np.sum(y))

chi2_values, p_values = chi2(X, y)
print ('chi square values')
print (chi2_values)
print ('p values')
print (p_values)

clf = RandomForestClassifier(n_estimators = 200, max_depth = 10, min_samples_leaf = 100,
                             random_state = 42, verbose = 2)
clf.fit(X, y)
print (clf.feature_importances_)
# feature_importances_
'''
