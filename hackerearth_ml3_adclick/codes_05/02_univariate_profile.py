import common_vars as c_vars
import pandas as pd
import numpy as np
from datetime import datetime

# df = pd.read_csv(c_vars.train_sample_file)
df = pd.read_csv(c_vars.train_split_train)
df = df[c_vars.header_useful]

# transformations on the columns
df.fillna(-999, inplace = True)
df.loc[:, 'datetime'] = df['datetime'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").day%7)
df.loc[:, 'browserid'] = df['browserid'].apply(lambda x: x if x not in c_vars.browserid_map 
                                                           else c_vars.browserid_map[x])
df['dummy'] = 1
# df.to_csv('../inputData/train_sample_uinvariate.csv', index = False)

df_list = []
cols_investigate = list(c_vars.header_useful)
cols_investigate.remove('click')
for col in cols_investigate:
    df_temp = df.pivot_table(index = col, columns = 'click', 
                             values = 'dummy', aggfunc = np.sum)
    df_temp.reset_index(inplace = True)
    df_temp['dimension'] = col
    # print (df_temp.columns.values)
    df_temp['row_count'] = df_temp[0] + df_temp[1]
    df_temp.rename(columns = {col:'dimension_value'}, inplace = True)
    df_list.append(df_temp)

    print ()
    print (col)
    print (df_temp.describe())

    df_temp['row_count'].quantile(np.arange(0, 1, 0.01)).to_csv('../analysis_graphs/quantile_' + str(col) + '.csv')

# pd.concat(df_list).to_csv('../analysis_graphs/train_train_univariate_profile.csv', index = False)