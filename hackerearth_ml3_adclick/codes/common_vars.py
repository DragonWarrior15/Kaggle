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

train_spilt_train_processed = '../inputData/train_train_processed'
train_spilt_val_processed = '../inputData/train_val_processed'
test_processed = '../inputData/test_processed'

# dictionaries for mapping similar items
browserid_map = {'Google Chrome':'Chrome',
                 'InternetExplorer':'IE','Internet Explorer':'IE','Edge':'IE',
                 'Mozilla Firefox':'Firefox','Mozilla':'Firefox'}

fillna_dict = {'datetime':'-999', 'siteid':-999, 'offerid':-999, 'category':-999, 'merchant':-999,
               'countrycode':'-999', 'browserid':'-999', 'devid':'-999', 'click':-999}

col_index_training = [13, 19, 14, 24, 18, 20, 12, 26, 11]

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