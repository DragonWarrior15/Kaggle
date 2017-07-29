header = ['ID', 'datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'click']
header_name_from_col = dict([[i, header[i]] for i in range(len(header))])
header_col_from_name = dict([[header[i], i] for i in range(len(header))])
header_useful = ['datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'click']

train_sample_file = '../inputData/train_sample.csv'
train_file = '../inputData/train.csv'
test_file = '../inputData/test.csv'

train_split_train = '../inputData/train_train.csv'
train_split_train_sample = '../inputData/train_train_sample.csv'
train_split_val = '../inputData/train_val.csv'

# dictionaries for mapping similar items
browserid_map = {'Google Chrome':'Chrome',
                 'InternetExplorer':'IE','Internet Explorer':'IE','Edge':'IE',
                 'Mozilla Firefox':'FireFox','Mozilla':'Firefox'}

fillna_dict = {'datetime':'-999', 'siteid':-999, 'offerid':-999, 'category':-999, 'merchant':-999,
               'countrycode':'-999', 'browserid':'-999', 'devid':'-999', 'click':-999}