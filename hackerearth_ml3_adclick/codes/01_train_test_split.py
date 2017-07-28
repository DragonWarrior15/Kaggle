import common_vars as c_vars
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv(c_vars.train_file).as_matrix()
# df = pd.read_csv(c_vars.train_sample_file).as_matrix()
df_train, df_val = train_test_split(df, test_size = 0.1, random_state = 42, stratify = df[:,-1])

df_train = pd.DataFrame(df_train, columns = c_vars.header)
df_val = pd.DataFrame(df_val, columns = c_vars.header)

df_train.to_csv(c_vars.train_split_train, index = False)
df_val.to_csv(c_vars.train_split_val, index = False)