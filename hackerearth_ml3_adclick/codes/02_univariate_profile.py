import common_vars as c_vars
import pandas as pd
from datetime import datetime

df = pd.read_csv(train_split_train)
df = df[[c_vars.header_useful]]
df.loc[:, 'datetime'] = df['datetime'].apply(lambda x: datetime.strptime("%Y-%m-%d %H:%M:%S").day)

df_list = []

for col in c_vars:
    df_temp = df.groupby([col])[[]]