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

# df = pd.read_csv(c_vars.train_sample_file)
df = pd.read_csv(c_vars.train_split_train_sample)
# df = pd.read_csv(c_vars.train_split_train)
df = df[c_vars.header_useful]

df.fillna(c_vars.fillna_dict, inplace = True)

df.loc[:, 'datetime_day'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
df.loc[:, 'datetime_hour'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
df = df.drop('datetime', axis=1)
df.loc[:, 'browserid'] = df['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           else c_vars.browserid_map[x])

c_vars.header_useful.remove('datetime')
c_vars.header_useful.remove('click')
c_vars.header_useful.append('datetime_day')
c_vars.header_useful.append('datetime_hour')

X = df[c_vars.header_useful].as_matrix()
y = df['click'].as_matrix()

with open('../analysis_graphs/label_encoder', 'rb') as f:
    label_encoder = pickle.load(f)
with open('../analysis_graphs/one_hot_encoding', 'rb') as f:
    ohe = pickle.load(f)

print (str(datetime.now()) + ' Label Encoding Started')
for i in range(len(label_encoder)):
    X[:,i] = label_encoder[i].transform(X[:,i])
print (str(datetime.now()) + ' Label Encoding Completed')

print (str(datetime.now()) + ' One Hot Encoding Started')
X_ohe = ohe.transform(X[:,c_vars.col_index_ohe])
print (str(datetime.now()) + ' One Hot Encoding Complete')

X = np.hstack((X[:,[i for i in range(len(c_vars.header_useful)) if i not in c_vars.col_index_ohe]], X_ohe))

print (X.shape, y.shape)
sm = SMOTE(random_state=42)
X, y = sm.fit_sample(X, y)

# save the X and y prepared
with open(c_vars.train_spilt_train_processed, 'wb') as f:
    pickle.dump([X, y], f)

df_test = pd.read_csv(c_vars.train_split_val)
df_test.fillna(c_vars.fillna_dict, inplace = True)
df_test.loc[:, 'datetime_day'] = df_test['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
df_test.loc[:, 'datetime_hour'] = df_test['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
df_test = df_test.drop('datetime', axis=1)
df_test.loc[:, 'browserid'] = df_test['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           else c_vars.browserid_map[x])
X_test = df_test[c_vars.header_useful].as_matrix()
y_test = df_test['click'].as_matrix()
for i in range(len(label_encoder)):
    X_test[:,i] = label_encoder[i].transform(X_test[:,i])
X_test_ohe = ohe.transform(X_test[:,c_vars.col_index_ohe])
X_test = np.hstack((X_test[:,[i for i in range(len(c_vars.header_useful)-1) if i not in c_vars.col_index_ohe]], X_test_ohe))

# save the X and y prepared
with open(c_vars.train_spilt_val_processed, 'wb') as f:
    pickle.dump([X_test, y_test], f)


# submit set
df_submit = pd.read_csv(c_vars.test_file)
df_submit.fillna(c_vars.fillna_dict, inplace = True)
df_submit.loc[:, 'datetime_day'] = df_submit['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
df_submit.loc[:, 'datetime_hour'] = df_submit['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
df_submit = df_submit.drop('datetime', axis=1)
df_submit.loc[:, 'browserid'] = df_submit['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           else c_vars.browserid_map[x])
X_submit = df_submit[c_vars.header_useful].as_matrix()
for i in range(len(label_encoder)):
    X_submit[:,i] = label_encoder[i].transform(X_submit[:,i])
X_submit_ohe = ohe.transform(X_submit[:,c_vars.col_index_ohe])
X_submit = np.hstack((X_submit[:,[i for i in range(len(c_vars.header_useful)-1) if i not in c_vars.col_index_ohe]], X_submit_ohe))

with open(c_vars.test_processed, 'wb') as f:
    pickle.dump(X_submit, f)