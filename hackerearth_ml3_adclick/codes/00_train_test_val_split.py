from datetime import datetime
import pickle

# file = '../inputData/train_sample.csv'
file = '../inputData/train.csv'

train_for_model = '../inputData/train_10to17_train.csv'
val_for_model = '../inputData/train_18to19_test.csv'
test_for_model = '../inputData/train_20_val.csv'


f_train = open(train_for_model, 'w')
f_val = open(val_for_model, 'w')
f_test = open(test_for_model, 'w')

columns = ['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click','year','month','day','hour','minute','second']
cols_to_int_dict = dict([[columns[i], i] for i in range(len(columns))])
cols_to_investigate = ['siteid','category','merchant','countrycode','browserid','devid','year','month','day','hour','minute','second']
target = ['click']

univariate_dict = {}
for col in cols_to_investigate:
    univariate_dict[col] = {}

firstLine = True
line_no = 0
with open(file, 'r') as f:
    # print (f.read())
    for line in f:
        line = line.strip().split(',')
        if not firstLine:
            datetime_data = datetime.strptime(line[cols_to_int_dict['datetime']], "%Y-%m-%d %H:%M:%S")

            if 10 <= datetime_data.day and datetime_data.day <= 17:
                file = f_train
            elif 18 <= datetime_data.day and datetime_data.day <= 19:
                file = f_val
            else:
                file = f_test
            
            file.write(','.join(line) + '\n')

        else:
            firstLine = False
            for file in [f_train, f_val, f_test]:
                file.write(','.join(line) + '\n')
