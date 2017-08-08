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

# with open('../analysis_graphs/ohe', 'rb') as f:
    # ohe = pickle.load(f)

with open('../analysis_graphs/standard_scaler', 'rb') as f:
    standard_scaler = pickle.load(f)

with open('../analysis_graphs/df_feature', 'rb') as f:
    df_feature = pickle.load(f)

for col in ['datetime', 'click', 'merchant', 'siteid', 'offerid', 'category']:
    c_vars.header_useful.remove(col)

# c_vars.header_useful.append('datetime_day')
# c_vars.header_useful.append('datetime_hour')

for col in ['merchant', 'siteid', 'offerid', 'category', 'countrycode', 'browserid', 'devid', 'datetime_hour', 'datetime_day'] +\
           ['countrycode_' + str(x) for x in ['merchant', 'siteid', 'offerid', 'category']] +\
           ['siteid_' + str(x) for x in ['merchant', 'offerid', 'category']]:
    for field in ['count', 'num_0', 'num_1', 'click_rate']:
        c_vars.header_useful.append(col + '_' + field)

print (c_vars.header_useful)

# df = pd.read_csv(c_vars.train_sample_file)
# df = pd.read_csv(c_vars.train_split_train)
def transformation_pipeline(df, preserve_id = False):
    df.fillna(c_vars.fillna_dict, inplace = True)

    df['datetime'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    df['datetime_day'] = df['datetime'].apply(lambda x: x.day%7)
    df['datetime_hour'] = df['datetime'].apply(lambda x: x.hour)
    df = df.drop('datetime', axis=1)

    for col in ['merchant', 'siteid', 'offerid', 'category']:
        df[col] = df[col].astype(np.int64)

    for col in ['merchant', 'siteid', 'offerid', 'category', 'countrycode', 'browserid', 'devid', 'datetime_hour', 'datetime_day']:
    # for col in ['merchant']:
        df = pd.merge(df, df_feature[col], how = 'left', on = col, suffixes = ('', ''))
        df.rename(columns = {'count':col+'_count', 'num_0':col+'_num_0', 
                             'num_1':col+'_num_1', 'click_rate':col+'_click_rate'}, 
                  inplace = True)
        # print(df['ID'].head(n = 20))

        # print (df.columns.tolist())

        if col in ['merchant', 'siteid', 'offerid', 'category']:
            for field in ['count', 'num_0', 'num_1', 'click_rate']:
                df[col + '_' + field].fillna(df_feature[col].loc[df_feature[col][col] == -99999, field].values[0], inplace = True)

    for col1, col2 in [['countrycode', x] for x in ['merchant', 'siteid', 'offerid', 'category']] +\
                      [['siteid', x] for x in ['merchant', 'offerid', 'category']]:
        col = col1 + '_' + col2
        df = pd.merge(df, df_feature[col], how = 'left', on = [col1, col2], suffixes = ('', ''))
        df.rename(columns = {'count':col+'_count', 'num_0':col+'_num_0', 
                             'num_1':col+'_num_1', 'click_rate':col+'_click_rate'}, 
              inplace = True)
        for field in ['count', 'num_0', 'num_1', 'click_rate']:
            df[col + '_' + field].fillna(df_feature[col].loc[(df_feature[col][col1] == -999999) & (df_feature[col][col2] == -999999),
                                         field].values[0], inplace = True)
    
    for col in df.columns.tolist():
        print (col, np.sum(df[col].isnull()), df[col].dtype)
    
    X = df[c_vars.header_useful].as_matrix()
    if preserve_id == True:
        df['ID'].to_csv(c_vars.test_processed_id, index = False, header = True)
        print (len(df['ID']))
    del df
    print (str(datetime.now()) + ' Label Encoding Started')
    for i in range(len(label_encoder)):
        X[:,i] = label_encoder[i].transform(X[:,i])
    print (str(datetime.now()) + ' Label Encoding Completed')

    X = X.astype(np.float64)

    print (str(datetime.now()) + ' Standard Scaler Started')
    X = standard_scaler.transform(X)
    print (str(datetime.now()) + ' Standard Scaler Completed')

    # print (str(datetime.now()) + ' OHE Started')
    # X_ohe = ohe.transform(X[:,[0,1,2,3,4]])
    # print (str(datetime.now()) + ' OHE Completed')

    # X = np.hstack((X[:,[i for i in range(len(c_vars.header_useful)) if i not in [0,1,2,3,4]]], X_ohe))
    # del X_ohe
    return (X)

'''
df = pd.read_csv(c_vars.train_split_train_sample)
X = transformation_pipeline(df)
y = df['click'].as_matrix()
print (X.shape, y.shape)
# save the X and y prepared
with open(c_vars.train_spilt_train_processed, 'wb') as f:
    pickle.dump([X, y], f)
'''
'''
df = pd.read_csv(c_vars.train_split_val)
X_unseen = transformation_pipeline(df)
y_unseen = df['click'].as_matrix()
# save the X and y prepared
with open(c_vars.train_spilt_val_processed, 'wb') as f:
    pickle.dump([X_unseen, y_unseen], f)
'''

# submit set
df_submit = pd.read_csv(c_vars.test_file)
X_submit = transformation_pipeline(df_submit, True)
with open(c_vars.test_processed, 'wb') as f:
    pickle.dump(X_submit, f)
