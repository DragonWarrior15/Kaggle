import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime
# from scipy import sparse
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# from imblearn.over_sampling import SMOTE

# from sklearn.feature_selection import chi2
# from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

c_vars.header_useful.remove('click')
print (c_vars.header_useful)

df = pd.read_csv(c_vars.train_file)
df = df[c_vars.header_useful]
df2 = pd.read_csv(c_vars.test_file)
df2 = df2[c_vars.header_useful]

df = pd.concat([df, df2])
print (len(df), len(df2))
print (df.columns.values, df2.columns.values)
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

print(df.columns.values)
c_vars.header_useful.remove('datetime')
# c_vars.header_useful.remove('click')
c_vars.header_useful.append('datetime_day')
c_vars.header_useful.append('datetime_hour')
print (c_vars.header_useful)

X = df[c_vars.header_useful].as_matrix()

print (str(datetime.now()) + ' Label Encoding Started')
label_encoder = [LabelEncoder() for _ in range(len(c_vars.header_useful))]
for i in range(len(label_encoder)):
    label_encoder[i].fit(X[:,i])
    # print (i, c_vars.header_useful[i], label_encoder[i].get_params(deep=True))
    X[:,i] = label_encoder[i].transform(X[:,i])
print (str(datetime.now()) + ' Label Encoding Completed')

print (str(datetime.now()) + ' One Hot Encoding Started')
cols_to_ohe = ['datetime', 'countrycode', 'browserid', 'devid']
ohe = OneHotEncoder(sparse = False)
ohe.fit(X[:,[4,5,6,7,8]])
# print (ohe.get_params(deep=True))
# X_ohe = ohe.transform(X[:,[0,5,6,7]])
print (str(datetime.now()) + ' One Hot Encoding Complete')

# save the label encoder and the one hot encoding to disk
with open('../analysis_graphs/label_encoder', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('../analysis_graphs/one_hot_encoding', 'wb') as f:
    pickle.dump(ohe, f)
