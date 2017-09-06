header = ['User_ID','Description','Browser_Used','Device_Used','Is_Response']
header_name_from_col = dict([[i, header[i]] for i in range(len(header))])
header_col_from_name = dict([[header[i], i] for i in range(len(header))])
header_useful = ['Description','Browser_Used','Device_Used','Is_Response']

# pre_path = '/data/vaibhav.ojha/'
pre_path = '../'

# train_sample_file = pre_path + 'inputs/train_sample.csv'
train_file = pre_path + 'inputs/train.csv'
test_file = pre_path + 'inputs/test.csv'

train_split_train = pre_path + 'inputs/train_train.csv'
train_split_train_sample = pre_path + 'inputs/train_train_sample.csv'
train_split_val = pre_path + 'inputs/train_val.csv'
train_split_train_0 = pre_path + 'inputs/train_train_0.csv'

browser_dict = {
    'Chrome'            : 'Chrome',
    'Edge'              : 'IE',
    'Firefox'           : 'Firefox',
    'Google Chrome'     : 'Chrome',
    'IE'                : 'IE',
    'Internet Explorer' : 'IE',
    'InternetExplorer'  : 'IE',
    'Mozilla'           : 'Firefox',
    'Mozilla Firefox'   : 'Firefox',
    'Opera'             : 'Others',
    'Safari'            : 'Others'
}

target_dict = {
    'happy' : 1,
    'not_happy' : 0
}