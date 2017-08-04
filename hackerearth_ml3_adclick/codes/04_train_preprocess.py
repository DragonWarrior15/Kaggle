import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
import pickle
import bisect

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)


# 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid'

# df = pd.read_csv(c_vars.train_file)
# df = pd.read_csv(c_vars.train_split_train_sample, nrows = 10000)
df = pd.read_csv(c_vars.train_split_train)
# df = df[c_vars.header_useful]

df.fillna(c_vars.fillna_dict, inplace = True)

# for col in df.columns.values:
    # print (col)
    # print (df[col].unique())
# print (df.dtypes)
# df['datetime'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df['datetime'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df['datetime_day'] = df['datetime'].apply(lambda x: x.day%7)
df['datetime_hour'] = df['datetime'].apply(lambda x: x.hour)
df = df.drop('datetime', axis=1)

for col in ['merchant', 'siteid', 'offerid', 'category']:
    df[col] = df[col].astype(np.int64)
# df.loc[:, 'browserid'] = df['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           # else c_vars.browserid_map[x])

threshold_dict = {'merchant':0.99, 'siteid':0.8, 'offerid':0.8, 'category':0.99}

# merchant, siteid, offerid, category
df_feature = {}
for col in ['merchant', 'siteid', 'offerid', 'category', 'countrycode', 'browserid', 'devid', 'datetime_hour', 'datetime_day']:
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

    if col in ['merchant', 'siteid', 'offerid', 'category']:
        df_temp_2 = df_temp.loc[df_temp['cumul_sum'] > threshold_dict[col] * len(df), :]
        df_temp = df_temp.loc[~(df_temp['cumul_sum'] > threshold_dict[col] * len(df)), :]
        df_temp = df_temp.append({col:-99999, 
                                  'count':np.sum(df_temp_2['count']), 
                                  'num_0':np.sum(df_temp_2['num_0']),
                                  'num_1':np.sum(df_temp_2['num_1'])},
                                 ignore_index = True)

    df_temp['click_rate'] = df_temp['num_1']/df_temp['count']
    df_temp.drop(['cumul_sum'], inplace = True, axis = 1)
    df_feature[col] = df_temp.loc[:,:]

    df = pd.merge(df, df_temp, how = 'left', on = col, suffixes = ('', ''))
    df.rename(columns = {'count':col+'_count', 'num_0':col+'_num_0', 
                         'num_1':col+'_num_1', 'click_rate':col+'_click_rate'}, 
              inplace = True)
    if col in ['merchant', 'siteid', 'offerid', 'category']:
        for field in ['count', 'num_0', 'num_1', 'click_rate']:
            df[col + '_' + field].fillna(df_feature[col].loc[df_temp[col] == -99999, field].values[0], inplace = True)
    # print (df.columns.values)

    # print (df)
# print (df)
print (df.columns.values)

for col in df.columns.tolist():
    print (col, np.sum(df[col].isnull()), df[col].dtype)

for col in ['datetime', 'click', 'merchant', 'siteid', 'offerid', 'category']:
    c_vars.header_useful.remove(col)

c_vars.header_useful.append('datetime_day')
c_vars.header_useful.append('datetime_hour')

for col in ['merchant', 'siteid', 'offerid', 'category', 'countrycode', 'browserid', 'devid', 'datetime_hour', 'datetime_day', 'datetime_minute']:
    for col2 in ['count', 'num_0', 'num_1', 'click_rate']:
        c_vars.header_useful.append(col + '_' + col2)

print (c_vars.header_useful)

X = df[c_vars.header_useful].as_matrix()
y = df['click'].as_matrix()

print (str(datetime.now()) + ' Label Encoding Started')
label_encoder = [LabelEncoder() for _ in range(3)]
for i in range(len(label_encoder)):
    label_encoder[i].fit(X[:,i])
    # print (i, c_vars.header_useful[i], label_encoder[i].get_params(deep=True))
    X[:,i] = label_encoder[i].transform(X[:,i])
print (str(datetime.now()) + ' Label Encoding Completed')

print (str(datetime.now()) + ' OHE Started')

ohe = OneHotEncoder(sparse = False)
ohe.fit(X[:,[0,1,2,3,4,5]])
X_ohe = ohe.transform(X[:,[0,1,2,3,4,5]])
print (str(datetime.now()) + ' OHE Completed')

X = np.hstack((X[:,[i for i in range(len(c_vars.header_useful)) if i not in [0,1,2,3,4,5]]], X_ohe))

'''
'''
# save the label encoder and the one hot encoding to disk
with open('../analysis_graphs/label_encoder', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('../analysis_graphs/ohe', 'wb') as f:
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
