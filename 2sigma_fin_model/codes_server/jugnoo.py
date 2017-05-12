import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 20000

print 'reading data'
df = pd.read_csv('data.csv')

col_list = list(df.columns)
col_list.remove('y')
col_list.remove('id')

df['count_null'] = df.isnull().sum(axis = 1)

for col in col_list:
  df[str(col + '_na')] = pd.isnull(df[col]).astype(int)
    
mean_values = df.mean(axis=0)
df.fillna(mean_values, inplace = True)

timestamp_list = sorted(np.unique(df['timestamp']))

for feature in col_list:
  corr_list = []
  for time in timestamp_list:
    temp_df = df[df['timestamp'] == time][[feature, 'y']]
    corr_list.append(np.corrcoef(temp_df[feature].values, temp_df['y'].values)[0, 1])
    # print(corr_list[-1])
  print (timestamp_list)
  print (corr_list)
  print ('saving figure : ' + feature)
  plt.clf()
  plt.grid(True)
  plt.scatter(timestamp_list, corr_list)
  plt.xlabel(feature)
  plt.ylabel('y')
  plt.savefig('images_time_corr/' + feature + '.png')
