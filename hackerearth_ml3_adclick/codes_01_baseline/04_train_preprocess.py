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
# df = pd.read_csv(c_vars.train_split_train_sample, nrows = 100000)
df = pd.read_csv(c_vars.train_split_train)
df = df[c_vars.header_useful]

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
c_vars.header_useful.remove('click')
c_vars.header_useful.append('datetime_day')
c_vars.header_useful.append('datetime_hour')
print (c_vars.header_useful)
for col in ['countrycode', 'browserid', 'devid', 'datetime_day', 'datetime_hour']:
    print (df[col].unique().tolist())

X = df[c_vars.header_useful].as_matrix()
y = df['click'].as_matrix()

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
X_ohe = ohe.transform(X[:,[4,5,6,7,8]])
print (str(datetime.now()) + ' One Hot Encoding Complete')
print (X.shape, X_ohe.shape)
# print (X.shape, X_ohe.shape)
# print (X, X_ohe)
X = np.hstack((X[:,[i for i in range(len(c_vars.header_useful)) if i not in [4,5,6,7,8]]], X_ohe))

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