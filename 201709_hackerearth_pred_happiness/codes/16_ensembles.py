import pandas as pd
import os

df_overall = pd.DataFrame()
counter = 0
for path, dirs, files in os.walk('../output/'):
    for file in files:
        df = pd.read_csv(os.path.join(path, file))
        if counter == 0:
            counter += 1
            df_overall = df.loc[:,:]
        else:
            df_overall = pd.merge(left = df_overall, right = df, how = 'inner', on = 'User_ID')

df_overall.to_csv('../analysis/ensembled.csv', index = False)
