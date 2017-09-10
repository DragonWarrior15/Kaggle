import common_vars as c_vars
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import re
import string
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import sys

reload(sys) 
sys.setdefaultencoding('utf8')
row_ind = 0
regex_punc = re.compile('[%s]' % re.escape(string.punctuation))
porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()
num_partitions = 2 #number of partitions to split dataframe
num_cores = 2 #number of cores on your machine

def cleaning_function(x):
    # x is a piece of text to be cleaned
    
    global row_ind
    if (row_ind + 1) % 1000 == 0:
        print (str(row_ind) + ' ' + str(datetime.now()))
    row_ind += 1
    
    x = unicode(x, errors='ignore')
    x = x.lower()
    # print (x)
    x = word_tokenize(x)
    
    # x = [regex_punc.sub(u'', token) for token in x if not regex_punc.sub(u'', token) == u'']
    x = [regex_punc.sub(u'', token) for token in x]
    x = list(filter(lambda x: x != u'', x))
    # x = [token for token in x if token not in stopwords.words('english')]
    x = list(filter(lambda x: x not in stopwords.words('english'), x))
    # x = [porter.stem(token) for token in x]
    # x = list(map(porter.stem, x))
    x = list(map(wordnet.lemmatize, x))
    
    x = ' '.join(x)
    return (x)

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def parallel_func_to_apply(data):
    data['Description_Clean'] = data['Description'].apply(lambda x: cleaning_function(x))
    return (data)

def main():
    df = pd.read_csv(c_vars.train_file)
    # df = pd.read_csv(c_vars.train_file_processed, encoding = "ISO-8859-1")
    # df = pd.read_csv(c_vars.train_file_processed)
    
    df.drop(['User_ID'], axis = 1, inplace = True)
    df['Is_Response'] = df['Is_Response'].apply(lambda x: 1 if x == 'happy' else 0)
    df['Browser_Used'] = df['Browser_Used'].apply(lambda x: c_vars.browser_dict[x])

    print ('Cleaning started at ' + str(datetime.now()))
    # df['Description'] = df['Description'].apply(lambda x: cleaning_function(x))
    df = parallelize_dataframe(df, parallel_func_to_apply)
    print ('Cleaning complete at ' + str(datetime.now()))
    df.to_csv(c_vars.train_file_processed, index = False)
    
    df['text_length'] = df['Description_Clean'].apply(lambda x: len(x))
    df['word_count'] = df['Description_Clean'].apply(lambda x: len(x.split(' ')))

    df_train, df_dev = train_test_split(df.as_matrix(), test_size = 0.2, random_state = 42)
    df_train = pd.DataFrame(df_train, columns = c_vars.header_useful + ['Description_Clean', 'text_length', 'word_count'])
    df_dev = pd.DataFrame(df_dev, columns = c_vars.header_useful + ['Description_Clean', 'text_length', 'word_count'])

    df_device = df_train.groupby(['Device_Used'])['Is_Response'].agg(['count', np.sum])
    df_device.reset_index(inplace = True)
    # print (df_device.columns.values)
    # df_device.columns = df_device.columns.get_level_values(0)
    df_device['target_rate'] = df_device['sum']/df_device['count']
    df_device = df_device[['Device_Used', 'target_rate']]

    df_train = pd.merge(df_train, df_device, how = 'left', on = 'Device_Used', suffixes = ('', ''))
    df_dev = pd.merge(df_dev, df_device, how = 'left', on = 'Device_Used', suffixes = ('', ''))

    X_train = df_train[['Description_Clean', 'text_length', 'word_count', 'target_rate']].as_matrix()
    X_dev = df_dev[['Description_Clean', 'text_length', 'word_count', 'target_rate']].as_matrix()
    y_train = df_train['Is_Response'].as_matrix().astype(np.int64)
    y_dev = df_dev['Is_Response'].as_matrix().astype(np.int64)

    # print ('X_train ' + str(X_train.shape))
    # print ('X_dev '   + str(X_dev.shape))
    # print ('y_train ' + str(y_train.shape))
    # print ('y_dev '   + str(y_dev.shape))

    # print (X_train[1,:])

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    # tfVect = CountVectorizer()
    # tfVect = TfidfVectorizer(min_df = 5, ngram_range = (2, 4))
    tfVect = TfidfVectorizer(min_df = 3, ngram_range = (1, 3))
    # tfVect = TfidfVectorizer(min_df = 5)

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

    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.ensemble import RandomForestClassifier

    for i in [0.1]:
    # for i in np.logspace(-3, 0, num = 0 + 3 + 1):
    # for i in np.logspace(0, 5, num = 0 + 5 + 1):
        # clf = RandomForestClassifier(max_depth=10, n_estimators=20, min_samples_split=5, min_samples_leaf=5, random_state = 42)
        clf = MultinomialNB(alpha = i)
        # clf = GaussianNB()
        # clf = SVC(C = i)
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_dev_tfidf)
        y_pred_proba = clf.predict_proba(X_dev_tfidf)[:,1]
        print (str(i) + ',' + str(accuracy_score(y_dev, y_pred)) + ',' + str(roc_auc_score(y_dev, y_pred)))

        df_dev['y_dev'] = y_dev
        df_dev['y_pred'] = y_pred
        df_dev['y_pred_proba'] = y_pred_proba
        df_dev.to_csv('../inputs/dev_analysis.csv', index = False)

        values = X_train_tfidf.max(0).toarray()[0]
        feature_names = np.array(tfVect.get_feature_names())
        features_series = pd.Series(values, index = feature_names)
        top_20 = features_series.nlargest(20)
        bot_20 = features_series.nsmallest(20)
        print ('bot_20')
        print (bot_20)
        print ('top_20')
        print (top_20)

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

if __name__ == '__main__':
    main()