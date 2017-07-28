header = ['ID', 'datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'click']
header_col_from_name = dict([[i, header[i]] for i in range(len(header))])
header_name_from_col = dict([[header[i], i] for i in range(len(header))])
header_useful = ['datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid']

train_sample_file = '../inputData/train_sample.csv'
train_file = '../inputData/train.csv'
test_file = '../inputData/test.csv'

train_split_train = '../inputData/train_train.csv'
train_split_val = '../inputData/train_val.csv'