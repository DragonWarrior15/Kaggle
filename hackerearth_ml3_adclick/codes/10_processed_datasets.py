import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

with open('../analysis_graphs/label_encoder', 'rb') as f:
    label_encoder = pickle.load(f)

with open('../analysis_graphs/df_feature', 'rb') as f:
    df_feature = pickle.load(f)

for col in ['datetime', 'click']:
    c_vars.header_useful.remove(col)

c_vars.header_useful.append('datetime_day')
c_vars.header_useful.append('datetime_hour')
c_vars.header_useful.append('datetime_minute')

for col in ['merchant', 'siteid', 'offerid', 'category', 'countrycode', 'browserid', 'devid', 'datetime_hour', 'datetime_day', 'datetime_minute']:
    for col2 in ['count', 'num_0', 'num_1', 'click_rate']:
        c_vars.header_useful.append(col + '_' + col2)

print (c_vars.header_useful)

# df = pd.read_csv(c_vars.train_sample_file)
# df = pd.read_csv(c_vars.train_split_train)
def transformation_pipeline(df):
    df.fillna(c_vars.fillna_dict, inplace = True)

    df.loc[:, 'datetime_day'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
    df.loc[:, 'datetime_hour'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    df.loc[:, 'datetime_minute'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").minute)
    df = df.drop('datetime', axis=1)

    for col in ['merchant', 'siteid', 'offerid', 'category']:
        df[col] = df[col].astype(np.int64)

    for col in ['merchant', 'siteid', 'offerid', 'category', 'countrycode', 'browserid', 'devid', 'datetime_hour', 'datetime_day', 'datetime_minute']:
    # for col in ['merchant']:
        df = pd.merge(df, df_feature[col], how = 'left', on = col, suffixes = ('', ''))
        df.rename(columns = {'count':col+'_count', 'num_0':col+'_num_0', 
                             'num_1':col+'_num_1', 'click_rate':col+'_click_rate'}, 
                  inplace = True)

    if col in ['merchant', 'siteid', 'offerid', 'category']:
        for field in ['count', 'num_0', 'num_1', 'click_rate']:
            df[col + '_' + field].fillna(df_feature[col].loc[df_temp[col] == 'LESS_FREQ', field].values[0], inplace = True)


    for index, col in enumerate(['siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid']):
        val_to_set = -9999 if col in ['siteid', 'offerid', 'category', 'merchant'] else 'other'
        df.loc[~df[col].isin(label_encoder[index].classes_), col] = 'other'
        print (col, len(df[~df[col].isin(label_encoder[index].classes_)]))
        # print (label_encoder[index].classes_)

    # print (label_encoder[3].classes_.shape)

    X = df[c_vars.header_useful].as_matrix()
    # y = df['click'].as_matrix()

    print (str(datetime.now()) + ' Label Encoding Started')
    for i in range(len(label_encoder)):
        print (i)
        X[:,i] = label_encoder[i].transform(X[:,i])
    print (str(datetime.now()) + ' Label Encoding Completed')

    return (X)


df = pd.read_csv(c_vars.train_split_train_sample)
X = transformation_pipeline(df)
y = df['click'].as_matrix()
print (X.shape, y.shape)
# save the X and y prepared
with open(c_vars.train_spilt_train_processed, 'wb') as f:
    pickle.dump([X, y], f)


'''
df_test = pd.read_csv(c_vars.train_split_val)
X_test = transformation_pipeline(df_test)
y_test = df_test['click'].as_matrix()
# save the X and y prepared
with open(c_vars.train_spilt_val_processed, 'wb') as f:
    pickle.dump([X_test, y_test], f)
'''

'''
# submit set
df_submit = pd.read_csv(c_vars.test_file)
X_submit = transformation_pipeline(df_submit)
with open(c_vars.test_processed, 'wb') as f:
    pickle.dump(X_submit, f)
'''