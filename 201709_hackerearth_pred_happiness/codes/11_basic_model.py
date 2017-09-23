import common_vars as c_vars
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re

df = pd.read_csv(c_vars.train_file)

df.drop(['User_ID'], axis = 1, inplace = True)
df['Is_Response'] = df['Is_Response'].apply(lambda x: 1 if x == 'happy' else 0)
df['Browser_Used'] = df['Browser_Used'].apply(lambda x: c_vars.browser_dict[x])

df['text_length'] = df['Description'].apply(lambda x: len(x))
df['word_count'] = df['Description'].apply(lambda x: len(x.split(' ')))

df_train, df_dev = train_test_split(df.as_matrix(), test_size = 0.2, random_state = 42)
df_train = pd.DataFrame(df_train, columns = c_vars.header_useful + ['text_length', 'word_count'])
df_dev = pd.DataFrame(df_dev, columns = c_vars.header_useful + ['text_length', 'word_count'])

df_device = df_train.groupby(['Device_Used'])['Is_Response'].agg(['count', np.sum])
df_device.reset_index(inplace = True)
# print (df_device.columns.values)
# df_device.columns = df_device.columns.get_level_values(0)
df_device['target_rate'] = df_device['sum']/df_device['count']
df_device = df_device[['Device_Used', 'target_rate']]

df_train = pd.merge(df_train, df_device, how = 'left', on = 'Device_Used', suffixes = ('', ''))
df_dev = pd.merge(df_dev, df_device, how = 'left', on = 'Device_Used', suffixes = ('', ''))

X_train = df_train[['Description', 'text_length', 'word_count', 'target_rate']].as_matrix()
X_dev = df_dev[['Description', 'text_length', 'word_count', 'target_rate']].as_matrix()
y_train = df_train['Is_Response'].as_matrix().astype(np.int64)
y_dev = df_dev['Is_Response'].as_matrix().astype(np.int64)

# print ('X_train ' + str(X_train.shape))
# print ('X_dev '   + str(X_dev.shape))
# print ('y_train ' + str(y_train.shape))
# print ('y_dev '   + str(y_dev.shape))

# print (X_train)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# tfVect = CountVectorizer()
tfVect = CountVectorizer(min_df = 5, ngram_range = (2, 4))
# tfVect = TfidfVectorizer(min_df = 5, ngram_range = (2, 4))
# tfVect = TfidfVectorizer()

tfVect.fit(X_train[:, 0])
X_train_tfidf = tfVect.transform(X_train[:, 0])
# X_train_tfidf = c_vars.add_feature(X_train_tfidf, X_train[:, 1].astype(np.float64))
# X_train_tfidf = c_vars.add_feature(X_train_tfidf, X_train[:, 2].astype(np.int64))
# X_train_tfidf = c_vars.add_feature(X_train_tfidf, X_train[:, 3].astype(np.int64))
X_dev_tfidf = tfVect.transform(X_dev[:, 0])
# X_dev_tfidf = c_vars.add_feature(X_dev_tfidf, X_dev[:, 1].astype(np.float64))
# X_dev_tfidf = c_vars.add_feature(X_dev_tfidf, X_dev[:, 2].astype(np.int64))
# X_dev_tfidf = c_vars.add_feature(X_dev_tfidf, X_dev[:, 3].astype(np.int64))

# print (X_train_tfidf)

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
'''
alpha  accuracy        roc
tfidf
0.001  0.84281494799   0.792332378184
0.01   0.852703223321  0.801827034115
0.1    0.854629510723  0.799457765859
1      0.79594195454   0.68603195022
10     0.679080518813  0.500599520384
100    0.678695261333  0.5
min_df = 5, tfidf
0.001  0.853088480801  0.80021666678
0.01   0.854115834082  0.801604922306
0.1    0.855913702324  0.804402684097
1      0.846025426994  0.778491719396
10     0.693591883909  0.523181454836
100    0.678695261333  0.5

SVC, tfidf
1.0    0.677539488892  0.509777041186
10.0   0.679465776294  0.514037436087
100.0  0.680878387055  0.516235677494

multinomialNB count vec, text length, word length, target rate
0.001  0.8518042892    0.82179040117
0.01   0.857326313086  0.827752707957
0.1    0.862719917812  0.834251803855
1.0    0.861820983691  0.830327336995
10.0   0.828945678695  0.745283319229
100.0  0.678695261333  0.5

multinomialNB tfidf min_df = 5 ngram_range=(2,4), LB 0.88379
0.001  0.877616540388  0.839543691442
0.01   0.878515474509  0.845467580524
0.1    0.87979966611   0.849465404025
1.0    0.830743546937  0.745029331313
10.0   0.679722614614  0.501598721023
100.0  0.678695261333  0.5
'''

for i in np.logspace(-3, 2, num = 2 + 3 + 1):
# for i in np.logspace(0, 5, num = 0 + 5 + 1):
    clf = MultinomialNB(alpha = i)
    # clf = SVC(C = i)
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_dev_tfidf)
    print (str(i) + ',' + str(accuracy_score(y_dev, y_pred)) + ',' + str(roc_auc_score(y_dev, y_pred)))

'''
clf = MultinomialNB(alpha = 0.1)
clf.fit(X_train_tfidf, y_train)

# predict on the submit set
df_submit = pd.read_csv(c_vars.test_file)

df_submit['text_length'] = df_submit['Description'].apply(lambda x: len(x))
df_submit['word_count'] = df_submit['Description'].apply(lambda x: len(x.split(' ')))
df_submit = pd.merge(df_submit, df_device, how = 'left', on = 'Device_Used', suffixes = ('', ''))
X_submit = df_submit[['Description', 'text_length', 'word_count', 'target_rate']].as_matrix()
X_submit_tfidf = tfVect.transform(X_submit[:, 0])
# X_submit_tfidf = c_vars.add_feature(X_submit_tfidf, X_submit[:, 1].astype(np.float64))
# X_submit_tfidf = c_vars.add_feature(X_submit_tfidf, X_submit[:, 2].astype(np.int64))
# X_submit_tfidf = c_vars.add_feature(X_submit_tfidf, X_submit[:, 3].astype(np.int64))

y_pred_submit = clf.predict(X_submit_tfidf)
df_submit['Is_Response'] = y_pred_submit
df_submit['Is_Response'] = df_submit['Is_Response'].apply(lambda x: 'happy' if x == 1 else 'not_happy')
df_submit[['User_ID', 'Is_Response']].to_csv('../output/submit_20170709_2306_0.1.csv', index = False)
'''