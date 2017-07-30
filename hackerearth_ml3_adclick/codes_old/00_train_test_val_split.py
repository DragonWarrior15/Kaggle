from datetime import datetime

# file = '../inputData/train_sample.csv'
file = '../inputData/train.csv'

train_for_model = '../inputData/train_10to18_train.csv'
test_for_model = '../inputData/train_19to20_test.csv'


f_train = open(train_for_model, 'w')
f_test = open(test_for_model, 'w')

columns = ['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click','year','month','day','hour','minute','second']
cols_to_int_dict = dict([[columns[i], i] for i in range(len(columns))])

firstLine = True
line_no = 0
with open(file, 'r') as f:
    # print (f.read())
    for line in f:
        line = line.strip().split(',')
        if not firstLine:
            datetime_data = datetime.strptime(line[cols_to_int_dict['datetime']], "%Y-%m-%d %H:%M:%S")

            if 10 <= datetime_data.day and datetime_data.day <= 18:
                file = f_train
            else:
                file = f_test
            
            file.write(','.join(line) + '\n')

        else:
            firstLine = False
            for file in [f_train, f_test]:
                file.write(','.join(line) + '\n')
