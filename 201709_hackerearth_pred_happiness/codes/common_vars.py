header = ['User_ID','Description','Browser_Used','Device_Used','Is_Response']
header_name_from_col = dict([[i, header[i]] for i in range(len(header))])
header_col_from_name = dict([[header[i], i] for i in range(len(header))])
header_useful = ['Description','Browser_Used','Device_Used','Is_Response']

# pre_path = '/data/vaibhav.ojha/'
pre_path = '../'

# train_sample_file = pre_path + 'inputs/train_sample.csv'
train_file = pre_path + 'inputs/train.csv'
test_file = pre_path + 'inputs/test.csv'
train_file_processed = pre_path + 'inputs/train_processed.csv'
test_file_processed = pre_path + 'inputs/test_processed.csv'

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

def add_feature(X, feature_to_add):
    '''
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    '''
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def get_param_space(param_dict):
    param_space = []
    param_list = sorted(list([k for k in param_dict]))
    param_to_int_dict = dict([[param_list[i], i] for i in range(len(param_list))])
    # print (param_to_int_dict)
    for param in param_list:
        curr_param_space_length = len(param_space)
        if (curr_param_space_length == 0):
            for i in range(len(param_dict[param])):
                param_space.append([param_dict[param][i]])
        else:
            for i in range(len(param_dict[param]) - 1):
                for j in range(curr_param_space_length):
                    param_space.append(list(param_space[j]) + [param_dict[param][i]])

            for i in range(curr_param_space_length):
                param_space[i].append(param_dict[param][-1])

    # print (param_space)
    param_space = sorted(param_space)
    return (param_space, param_to_int_dict)
