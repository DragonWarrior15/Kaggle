import common_vars as c_vars
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
import re
import string
from multiprocessing import Pool

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import sys
import gc

<<<<<<< HEAD
from scipy import sparse
=======
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from scipy.sparse import hstack, csr_matrix, coo_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.utils import safe_mask

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

<<<<<<< HEAD
# codes for compatibility between python 2 and 3
try:
    reload(sys)
except NameError:
    pass
try:
    sys.setdefaultencoding('utf8')
except AttributeError:
    pass

=======
reload(sys)
sys.setdefaultencoding('utf8')
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
row_ind = 0

regex_punc = re.compile('[%s]' % re.escape(string.punctuation))
regex_nt = re.compile('nt')
regex_horrible = re.compile('["]+')
regex_hyphen = re.compile('[-]+(st|nd|rd|th)?')

stopwords_set = set(stopwords.words('english')) - set(['but','if','as','until','while','of','against','again','then','once','no','nor','not','only','too','very','should','ain','aren','couldn','didn','doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won',])
porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()
num_partitions = 2 #number of partitions to split dataframe
num_cores = 2 #number of cores on your machine

def cleaning_function(x):
    # x is a piece of text to be cleaned

<<<<<<< HEAD
    try:
        x = unicode(x, errors='ignore')
    except NameError:
        pass
=======
    x = unicode(x, errors='ignore')
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
    x = x.lower()
    # print (x)
    x = regex_horrible.sub(u' horrible ', x)
    x = regex_hyphen.sub(u'', x)
    x = word_tokenize(x)
    
    # x = [regex_punc.sub(u'', token) for token in x if not regex_punc.sub(u'', token) == u'']
    x = [regex_punc.sub(u'', token) for token in x]
    x = [regex_nt.sub(u'not', token) for token in x]
    x = list(filter(lambda x: x != u'', x))
    # x = [token for token in x if token not in stopwords.words('english')]
    x = list(filter(lambda word: word not in stopwords_set, x))
    # x = [porter.stem(token) for token in x]
    # x = list(map(porter.stem, x))
    # x = list(map(wordnet.lemmatize, x))
    x = list(map(snowball.stem, x))
    
    x = ' '.join(x)
    return (x)

def pos_tag_adj_extract(x):
    x = pos_tag(word_tokenize(x))
    x = list(filter(lambda word: 'JJ' in word[1], x))
    x = ' '.join(list(map(lambda word: word[0], x)))
    return (x)

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return (df)

def parallel_func_to_apply(data):
    data['Description_Clean'] = data['Description'].apply(lambda x: cleaning_function(x))
    data['Description_Clean_Adj'] = data['Description_Clean'].apply(lambda x: pos_tag_adj_extract(x))
    return (data)


def main():
    
    # df = pd.read_csv(c_vars.train_file)
    df = pd.read_csv(c_vars.train_file_processed, encoding = "ISO-8859-1")
    df['Description_Clean'].fillna('', inplace = True, axis = 0)
    df['Description_Clean_Adj'].fillna('', inplace = True, axis = 0)
    '''
    df.drop(['User_ID'], axis = 1, inplace = True)
    df['Is_Response'] = df['Is_Response'].apply(lambda x: 1 if x == 'happy' else 0)
    df['Browser_Used'] = df['Browser_Used'].apply(lambda x: c_vars.browser_dict[x])

    # print ('Cleaning started at ' + str(datetime.now()))
    df = parallelize_dataframe(df, parallel_func_to_apply)
    # print ('Cleaning complete at ' + str(datetime.now()))
    df.to_csv(c_vars.train_file_processed, index = False)
    
    df_submit = pd.read_csv(c_vars.test_file)
    df_submit['Browser_Used'] = df_submit['Browser_Used'].apply(lambda x: c_vars.browser_dict[x])
    df_submit = parallelize_dataframe(df_submit, parallel_func_to_apply)
    df_submit.to_csv(c_vars.test_file_processed, index = False)
    # sys.exit()
    '''
    df['text_length'] = df['Description_Clean'].apply(lambda x: len(x))
    df['word_count'] = df['Description_Clean'].apply(lambda x: len(x.split(' ')))

    # create more copies of the unhappy/bad reviews to identify those words
    # df = pd.concat([df, df.loc[df['Is_Response'] == 0,], df.loc[df['Is_Response'] == 0,]])

<<<<<<< HEAD
    df_train, df_dev = train_test_split(df.as_matrix(), test_size = 0.2, random_state = 442)
=======
    df_train, df_dev = train_test_split(df.as_matrix(), test_size = 0.1, random_state = 442)
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
    df_train = pd.DataFrame(df_train, columns = c_vars.header_useful + ['Description_Clean', 'Description_Clean_Adj', 'text_length', 'word_count'])
    df_dev = pd.DataFrame(df_dev, columns = c_vars.header_useful + ['Description_Clean', 'Description_Clean_Adj', 'text_length', 'word_count'])

    df_device = df_train.groupby(['Device_Used'])['Is_Response'].agg(['count', np.sum])
    df_device.reset_index(inplace = True)
    # print (df_device.columns.values)
    # df_device.columns = df_device.columns.get_level_values(0)
    df_device['target_rate'] = df_device['sum']/df_device['count']
    df_device = df_device[['Device_Used', 'target_rate']]

    df_train = pd.merge(df_train, df_device, how = 'left', on = 'Device_Used', suffixes = ('', ''))
    df_dev = pd.merge(df_dev, df_device, how = 'left', on = 'Device_Used', suffixes = ('', ''))

    X_train = df_train[['Description_Clean', 'Description_Clean_Adj', 'text_length', 'word_count', 'target_rate']].as_matrix()
    X_dev = df_dev[['Description_Clean', 'Description_Clean_Adj', 'text_length', 'word_count', 'target_rate']].as_matrix()
    y_train_tfidf = df_train['Is_Response'].as_matrix().astype(np.int64)
    y_dev_tfidf = df_dev['Is_Response'].as_matrix().astype(np.int64)

    # print ('X_train ' + str(X_train.shape))
    # print ('X_dev '   + str(X_dev.shape))
    # print ('y_train ' + str(y_train.shape))
    # print ('y_dev '   + str(y_dev.shape))

    # print (X_train[1,:])

    # tfVect = TfidfVectorizer(min_df = 5, ngram_range = (2, 4))
    # tfVect = TfidfVectorizer(min_df = 3, ngram_range = (1, 3))
<<<<<<< HEAD
    tfVect1 = TfidfVectorizer(max_features=3600, ngram_range = (1,1))
    tfVect2 = TfidfVectorizer(max_features=4000, ngram_range = (2,2))
    tfVect3 = TfidfVectorizer(max_features=3500, ngram_range = (1,1))
    # countVect = CountVectorizer()
=======
    tfVect1 = TfidfVectorizer(max_features = 200, ngram_range = (1,1), min_df = 50)
    tfVect2 = TfidfVectorizer(max_features = 400, ngram_range = (2,2), min_df = 50)
    tfVect3 = TfidfVectorizer(max_features = 200, ngram_range = (1,1), min_df = 50)
    countVect = CountVectorizer()
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
    # tfVect = TfidfVectorizer(min_df = 5)

    tfVect1.fit(X_train[:, 0])
    tfVect2.fit(X_train[:, 0])
    tfVect3.fit(X_train[:, 1])
    # countVect.fit(X_train[:, 1])
<<<<<<< HEAD
    X_train_tfidf = hstack((tfVect1.transform(X_train[:, 0]), tfVect2.transform(X_train[:, 0]), tfVect3.transform(X_train[:, 1])))
=======
    # X_train_tfidf = hstack((tfVect2.transform(X_train[:, 0]), tfVect3.transform(X_train[:, 1])))
    X_train_tfidf = hstack((tfVect1.transform(X_train[:, 0]), tfVect2.transform(X_train[:, 0]), tfVect3.transform(X_train[:, 1])))
    # X_train_tfidf = hstack((tfVect1.transform(X_train[:, 0]), tfVect2.transform(X_train[:, 0]), tfVect3.transform(X_train[:, 1]), countVect.transform(X_train[:, 1])))
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
    # truncatedsvd = TruncatedSVD(n_components = 500, random_state = 42)
    # truncatedsvd.fit(X_train_tfidf)
    # X_train_tfidf = truncatedsvd.transform(X_train_tfidf)
    # X_train_tfidf = c_vars.add_feature(X_train_tfidf, X_train[:, 2].astype(np.float64))
    # X_train_tfidf = c_vars.add_feature(X_train_tfidf, X_train[:, 3].astype(np.int64))
    # X_train_tfidf = c_vars.add_feature(X_train_tfidf, X_train[:, 4].astype(np.int64))
    
    # X_dev_tfidf = hstack((tfVect2.transform(X_dev[:, 0]), tfVect3.transform(X_dev[:, 1])))
    X_dev_tfidf = hstack((tfVect1.transform(X_dev[:, 0]), tfVect2.transform(X_dev[:, 0]), tfVect3.transform(X_dev[:, 1])))
    # X_dev_tfidf = hstack((tfVect1.transform(X_dev[:, 0]), tfVect2.transform(X_dev[:, 0]), tfVect3.transform(X_dev[:, 1]), countVect.transform(X_dev[:, 1])))
    # X_dev_tfidf = truncatedsvd.transform(X_dev_tfidf)
    # X_dev_tfidf = c_vars.add_feature(X_dev_tfidf, X_dev[:, 2].astype(np.float64))
    # X_dev_tfidf = c_vars.add_feature(X_dev_tfidf, X_dev[:, 3].astype(np.int64))
    # X_dev_tfidf = c_vars.add_feature(X_dev_tfidf, X_dev[:, 4].astype(np.int64))

    # scaler = StandardScaler(with_mean = False)
    # scaler.fit(X_train_tfidf)

    # X_train_tfidf = scaler.transform(X_train_tfidf)
    # X_dev_tfidf = scaler.transform(X_dev_tfidf)

    # print (X_train_tfidf)

    param_dict = {'max_depth' : [5, 18, 15],
                  'n_estimators' : [120, 300, 500],
                  'min_samples_split' : [2, 5, 10],
                  'min_samples_leaf' : [1, 2, 5, 10]}
    param_dict = {
              'colsample_bytree' : [0.6, 0.8],
              'learning_rate' : [0.01, 0.025, 0.5],
              'max_depth' : [3, 5],
              'min_child_weight' : [5],
              'n_estimators' : [10],
              'reg_alpha' : [0],
              'reg_lambda' : [10],
              'subsample' : [0.8, 1]
              }
    param_dict = {
            'penalty' : ['l2'],
            'C' : [1.01]
    }
    param_space, param_to_int_dict = c_vars.get_param_space(param_dict)

    for param_list in param_space:
    # for i in [0.1]:
<<<<<<< HEAD
    # for i in [1]:
    # for i in [1.01]:
    # for i in np.logspace(-3, 0, num = 0 + 3 + 1):
    # for i in np.logspace(-1, 3, num = 1 + 3 + 1):
        print (param_list)
        # clf = RandomForestClassifier(max_depth=param_list[param_to_int_dict['max_depth']], 
                             # n_estimators=param_list[param_to_int_dict['n_estimators']],
                             # min_samples_split=param_list[param_to_int_dict['min_samples_split']],
                             # min_samples_leaf=param_list[param_to_int_dict['min_samples_leaf']],
                             # random_state = 42)
        # clf = XGBClassifier(colsample_bytree      = param_list[param_to_int_dict['colsample_bytree']],
                            # learning_rate         = param_list[param_to_int_dict['learning_rate']],
                            # max_depth             = param_list[param_to_int_dict['max_depth']],
                            # min_child_weight      = param_list[param_to_int_dict['min_child_weight']],
                            # n_estimators          = param_list[param_to_int_dict['n_estimators']],
                            # reg_alpha             = param_list[param_to_int_dict['reg_alpha']],
                            # reg_lambda            = param_list[param_to_int_dict['reg_lambda']],
                            # subsample             = param_list[param_to_int_dict['subsample']])
=======
    # for i in [0.000001, 0.00001, 0.0001, 0.001]:
    for i in [1.01]:
    # for i in np.logspace(-3, 0, num = 0 + 3 + 1):
    # for i in np.logspace(-1, 3, num = 1 + 3 + 1):
        kf = StratifiedKFold(n_splits = 4, shuffle = True)
        # clf = RandomForestClassifier(max_depth=8, n_estimators=100, min_samples_split=5, min_samples_leaf=5, random_state = 42)
        # clf = XGBClassifier(
                    # colsample_bytree      = 0.6,
                    # learning_rate         = 0.05,
                    # max_depth             = 4,
                    # min_child_weight      = 1,
                    # n_estimators          = 20,
                    # reg_alpha             = 0,
                    # reg_lambda            = 10,
                    # subsample             = 0.8,
                    # )
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
        # clf = GradientBoostingClassifier(
                # max_depth        = 7,
                # n_estimators     = 40,
                # learning_rate    = 1,
                # random_state = 42)
        # clf = MLPClassifier(activation         = 'logistic',
                            # hidden_layer_sizes = (200, 50, 10),
                            # learning_rate      = 'invscaling',
                            # max_iter           = 200,
                            # solver             = 'adam',
                            # random_state = 42)
        # clf = MultinomialNB(alpha = i)
        # clf = GaussianNB()
        # clf = SVC(C = i)
<<<<<<< HEAD
        clf = LogisticRegression(penalty = param_list[param_to_int_dict['penalty']],
                                 C = param_list[param_to_int_dict['C']])

        # if type(X_train_tfidf) is not np.ndarray:
            # X_train_tfidf = X_train_tfidf.toarray()
            # X_dev_tfidf = X_dev_tfidf.toarray()
=======
        # clf = LogisticRegression(penalty = 'l2', C = i)
        clf = CatBoostClassifier(iterations=50, learning_rate=0.03, depth=4, loss_function='Logloss', class_weights = [1/0.67,1])
        if type(X_train_tfidf) is not np.ndarray:
            X_train_tfidf = X_train_tfidf.toarray()
            X_dev_tfidf = X_dev_tfidf.toarray()
        kf_index = 0
        for train_indices, test_indices in kf.split(X_train_tfidf, y_train_tfidf):
            kf_index += 1
            # print (kf_index)
            if type(X_train_tfidf) is np.ndarray:
                X_train, X_val = X_train_tfidf[train_indices], X_train_tfidf[test_indices]
            else:
                X_train, X_val = csr_matrix(X_train_tfidf)[safe_mask(X_train_tfidf, train_indices), :], csr_matrix(X_train_tfidf)[safe_mask(X_train_tfidf, test_indices), :]
            y_train, y_val = y_train_tfidf[train_indices], y_train_tfidf[test_indices]

            clf.fit(X_train, y_train)
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b

            # val set
            y_pred = clf.predict(X_val)
            y_pred_proba = clf.predict_proba(X_val)[:,1]
            y_pred_train = clf.predict(X_train)
            y_pred_proba_train = clf.predict_proba(X_train)[:,1]

<<<<<<< HEAD
        print ('Train, ' + str(accuracy_score(y_train, y_pred_train)) + ',' + str(roc_auc_score(y_train, y_pred_train)))
        print ('Test, ' + str(accuracy_score(y_dev, y_pred)) + ',' + str(roc_auc_score(y_dev, y_pred)))
=======
            print ('Train ' + 'CV ' + str(kf_index) + ' ,' + str(i) + ' ,' + str(accuracy_score(y_train, y_pred_train)) + ',' + str(roc_auc_score(y_train, y_pred_proba_train)))
            print ('Val   ' + 'CV ' + str(kf_index) + ' ,' + str(i) + ' ,' + str(accuracy_score(y_val, y_pred)) + ',' + str(roc_auc_score(y_val, y_pred_proba)))

            # dev set
            y_pred = clf.predict(X_dev_tfidf)
            y_pred_proba = clf.predict_proba(X_dev_tfidf)[:,1]

            print ('Dev   ' + 'CV ' + str(kf_index) + ' ,' + str(i) + ' ,' + str(accuracy_score(y_dev_tfidf, y_pred)) + ',' + str(roc_auc_score(y_dev_tfidf, y_pred_proba)))
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b

        
        df_dev['y_dev'] = y_dev
        df_dev['y_pred'] = y_pred
        df_dev['y_pred_proba'] = y_pred_proba
        df_dev.to_csv('../analysis/dev_analysis.csv', index = False)
        '''
        values = X_train_tfidf.max(0).toarray()[0]
        # feature_names = np.hstack((np.array(tfVect1.get_feature_names()), np.array(tfVect2.get_feature_names()), np.array(tfVect3.get_feature_names())))
        feature_names = np.hstack((np.array(tfVect1.get_feature_names()), np.array(tfVect2.get_feature_names())))
        print (feature_names.shape)
        features_series = pd.DataFrame(values, index = feature_names)
        features_series['coefs'] = clf.coef_.reshape(-1, 1)
        features_series.to_csv('../analysis/features_series.csv')
        '''
        '''
        f = open('../analysis/corr.csv', 'w')
        for i in range(X_train_tfidf.shape[1]):
            f.write (str(features_series.index[i]) + ',' + str(features_series[i]) + ',' + str(i) + ',' + str(pearsonr(X_train_tfidf.toarray()[:,i], y_train)[0]) + '\n')
        f.close()
        '''
        # top_20 = features_series.nlargest(20)
        # bot_20 = features_series.nsmallest(20)
        # print ('bot_20')
        # print (bot_20)
        # print ('top_20')
        # print (top_20)

<<<<<<< HEAD
    
    # delete training data to clean up memory
    del X_train
    del X_train_tfidf
    del df_train
    del X_dev
    del X_dev_tfidf
    del df_dev
    del y_train
    del y_dev
    del y_pred
    del y_pred_proba
    del y_pred_train
    del y_pred_proba_train
    gc.collect()

=======
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
    
    # delete training data to clean up memory
    del X_train
    del X_train_tfidf
    del df_train
    del X_dev
    del X_dev_tfidf
    del df_dev
    del y_train
    del y_pred
    del y_pred_proba
    del y_pred_train
    del y_pred_proba_train
    gc.collect()

    '''
    # predict on the submit set
<<<<<<< HEAD
    df_submit = pd.read_csv(c_vars.test_file_processed, encoding = "ISO-8859-1")
=======
    df_submit = pd.read_csv(c_vars.test_file_processed)
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
    df_submit['Description_Clean'].fillna('', inplace = True, axis = 0)
    df_submit['Description_Clean_Adj'].fillna('', inplace = True, axis = 0)

    df_submit['text_length'] = df_submit['Description_Clean'].apply(lambda x: len(x))
    df_submit['word_count'] = df_submit['Description_Clean'].apply(lambda x: len(x.split(' ')))

    df_submit = pd.merge(df_submit, df_device, how = 'left', on = 'Device_Used', suffixes = ('', ''))
    X_submit = df_submit[['Description_Clean', 'Description_Clean_Adj', 'text_length', 'word_count', 'target_rate']].as_matrix()
<<<<<<< HEAD
    # X_submit_tfidf = hstack((tfVect1.transform(X_submit[:, 0]), tfVect2.transform(X_submit[:, 0]), tfVect3.transform(X_submit[:, 1])))
    X_submit_tfidf = hstack((tfVect1.transform(X_submit[:, 0]), tfVect2.transform(X_submit[:, 0]), tfVect3.transform(X_submit[:, 1])))
=======
    X_submit_tfidf = hstack((tfVect1.transform(X_submit[:, 0]), tfVect2.transform(X_submit[:, 0])))
    # X_submit_tfidf = hstack((tfVect1.transform(X_submit[:, 0]), tfVect2.transform(X_submit[:, 0]), tfVect3.transform(X_submit[:, 1])))
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b
    # X_submit_tfidf = X_submit_tfidf.toarray()
    # X_submit_tfidf = truncatedsvd.transform(X_submit_tfidf)
    # X_submit_tfidf = tfVect.transform(X_submit[:, 0])
    # X_submit_tfidf = c_vars.add_feature(X_submit_tfidf, X_submit[:, 1].astype(np.float64))
    # X_submit_tfidf = c_vars.add_feature(X_submit_tfidf, X_submit[:, 2].astype(np.int64))
    # X_submit_tfidf = c_vars.add_feature(X_submit_tfidf, X_submit[:, 3].astype(np.int64))

    y_pred_submit = clf.predict(X_submit_tfidf)
    df_submit['Is_Response'] = y_pred_submit
    df_submit['Is_Response'] = df_submit['Is_Response'].apply(lambda x: 'happy' if x == 1 else 'not_happy')
<<<<<<< HEAD
    df_submit[['User_ID', 'Is_Response']].to_csv('../output/submit_20170917_2230_1_lr.csv', index = False)
    
=======
    df_submit[['User_ID', 'Is_Response']].to_csv('../output/submit_20170922_1430_1_lr.csv', index = False)
    '''
>>>>>>> 046558350d3fe7e9a4c1dfba22e98440ad2f771b

if __name__ == '__main__':
    main()