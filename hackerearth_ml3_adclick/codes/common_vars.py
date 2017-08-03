header = ['ID', 'datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'click']
header_name_from_col = dict([[i, header[i]] for i in range(len(header))])
header_col_from_name = dict([[header[i], i] for i in range(len(header))])
header_useful = ['datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'click']

pre_path = '/data/vaibhav.ojha/'
# pre_path = '../'

train_sample_file = pre_path + 'inputData/train_sample.csv'
train_file = pre_path + 'inputData/train.csv'
test_file = pre_path + 'inputData/test.csv'

train_split_train = pre_path + 'inputData/train_train.csv'
train_split_train_sample = pre_path + 'inputData/train_train_sample.csv'
train_split_val = pre_path + 'inputData/train_val.csv'
train_split_train_0 = pre_path + 'inputData/train_train_0.csv'

train_spilt_train_processed = pre_path + 'inputData/train_train_processed'
train_spilt_val_processed = pre_path + 'inputData/train_val_processed'
test_processed = pre_path + 'inputData/test_processed'

# dictionaries for mapping similar items
browserid_map = {'Google Chrome':'Chrome',
                 'InternetExplorer':'IE','Internet Explorer':'IE','Edge':'IE',
                 'Mozilla Firefox':'Firefox','Mozilla':'Firefox'}

fillna_dict = {'datetime':'-999', 'siteid':-999, 'offerid':-999, 'category':-999, 'merchant':-999,
               'countrycode':'-999', 'browserid':'-999', 'devid':'-999', 'click':0}

# col_index_training = [13, 19, 14, 24, 18, 20, 12, 26, 11]
col_index_training = [18,11,24,7,17,10,19,31,5,6,4,23,29,16,25,30,21,20,60,9,28,59,8,3,61,55,22,56,54,52,15,38,62,58,2,53,51,27,26,39,0,37,50,36,1,35,49,14,12,48,47,32,13,57,33,40,34,45,46,41,43,42,44]
col_index_ohe = [0,1,2,3,4]
num_features_for_model = 63
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