import common_vars as c_vars
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd

# df = pd.read_csv(c_vars.train_file).as_matrix()
df = pd.read_csv(c_vars.train_file)
df['datetime_day'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
df['datetime_day'] = df['datetime_day'].apply(lambda x: x.day)
df_val = df.loc[df['datetime_day'] >= 18, :]
df_train = df.loc[df['datetime_day'] < 18, :]
df_val = df_val.drop('datetime_day', axis=1)
df_train = df_train.drop('datetime_day', axis=1)
# df = pd.read_csv(c_vars.train_sample_file).as_matrix()
# df_train, df_val = train_test_split(df, test_size = 0.1, random_state = 42, stratify = df[:,-1])

df_train = pd.DataFrame(df_train, columns = c_vars.header)
df_val = pd.DataFrame(df_val, columns = c_vars.header)

df_train.to_csv(c_vars.train_split_train, index = False)
df_val.to_csv(c_vars.train_split_val, index = False)

df_train_1 = df_train.loc[df_train['click'] == 1, :]
df_train_0 = df_train.loc[df_train['click'] == 0, :]

pd.concat([df_train_0.sample(len(df_train_1)), df_train_1]).to_csv(c_vars.train_split_train_sample, index = False)
df_train_0.to_csv(c_vars.train_split_train_0, index = False)

'''
# split the train set into a smaller file for feature selection
df_train = df_train.as_matrix()
_, df_val = train_test_split(df_train, test_size = 0.1, random_state = 42, stratify = df_train[:,-1])
df_val = pd.DataFrame(df_val, columns = c_vars.header)
df_val.to_csv(c_vars.train_split_train_sample, index = False)
'''