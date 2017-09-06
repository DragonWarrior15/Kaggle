import common_vars as cvars
import pandas as pd

df = pd.read_csv(cvars.train_file)
print (df.columns.values)
print (len(df))

df['Browser_Used'] = df['Browser_Used'].apply(lambda x: cvars.browser_dict[x])

for col in df.columns.values:
    print (col, len(df[col].unique()))

print (df.groupby(['Is_Response']).count().reset_index())
print (df.groupby(['Browser_Used', 'Is_Response']).count().reset_index())
print (df.groupby(['Device_Used', 'Is_Response']).count().reset_index())
print (df.groupby(['Device_Used', 'Browser_Used', 'Is_Response']).count().reset_index())

df.groupby(['Device_Used', 'Browser_Used', 'Is_Response']).count().reset_index().to_csv('temp.csv', index = False)

print (df.isnull().sum(axis = 0))