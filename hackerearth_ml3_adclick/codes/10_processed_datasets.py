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
with open('../analysis_graphs/one_hot_encoding', 'rb') as f:
    ohe = pickle.load(f)
with open('../analysis_graphs/df_feature', 'rb') as f:
    df_feature = pickle.load(f)

for col in ['datetime', 'click', 'merchant', 'siteid', 'offerid', 'category']:
    c_vars.header_useful.remove(col)

c_vars.header_useful.append('datetime_day')
c_vars.header_useful.append('datetime_hour')

for col in ['merchant', 'siteid', 'offerid', 'category']:
    for col2 in ['count', 'num_0', 'num_1', 'click_rate']:
        c_vars.header_useful.append(col2 + '_' + col)

print (c_vars.header_useful)

# df = pd.read_csv(c_vars.train_sample_file)
# df = pd.read_csv(c_vars.train_split_train)
def transformation_pipeline(df):
    df.fillna(c_vars.fillna_dict, inplace = True)

    df.loc[:, 'datetime_day'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
    df.loc[:, 'datetime_hour'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    df = df.drop('datetime', axis=1)
    df.loc[:, 'browserid'] = df['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                               else c_vars.browserid_map[x])

    for col in ['merchant', 'siteid', 'offerid', 'category']:
    # for col in ['merchant']:
        df = pd.merge(df, df_feature[col], how = 'left', on = col, suffixes = ('', ''))
        df.rename(columns = {'count':'count_'+col, 'num_0':'num_0_'+col, 
                             'num_1':'num_1_'+col, 'click_rate':'click_rate_'+col}, 
                  inplace = True)
        for field in ['count', 'num_0', 'num_1', 'click_rate']:
            df[field + '_' + col].fillna(df_feature[col].loc[df_feature[col][col] == 'LESS_FREQ', field].values[0], inplace = True)


    X = df[c_vars.header_useful].as_matrix()
    # y = df['click'].as_matrix()

    print (str(datetime.now()) + ' Label Encoding Started')
    for i in range(len(label_encoder)):
        X[:,i] = label_encoder[i].transform(X[:,i])
    print (str(datetime.now()) + ' Label Encoding Completed')

    print (str(datetime.now()) + ' One Hot Encoding Started')
    X_ohe = ohe.transform(X[:,c_vars.col_index_ohe])
    print (str(datetime.now()) + ' One Hot Encoding Complete')

    X = np.hstack((X[:,[i for i in range(len(c_vars.header_useful)) if i not in c_vars.col_index_ohe]], X_ohe))

    return (X)


df = pd.read_csv(c_vars.train_split_train_sample)
X = transformation_pipeline(df)
y = df['click'].as_matrix()
print (X.shape, y.shape)
sm = SMOTE(random_state=42)
X, y = sm.fit_sample(X, y)
print (X.shape, y.shape)
_, X, _, y = train_test_split(X, y, test_size = 0.1, random_state = 42)
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