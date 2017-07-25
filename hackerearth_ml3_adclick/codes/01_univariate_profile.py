from datetime import datetime
import pickle

# file = '../inputData/train_sample.csv'
# file = '../inputData/train.csv'
file = '../inputData/train_10to17_train.csv'

columns = ['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click','year','month','day','hour','minute','second']
cols_to_int_dict = dict([[columns[i], i] for i in range(len(columns))])
cols_to_investigate = ['siteid','category','merchant','countrycode','browserid','devid','year','month','day','hour','minute','second']
target = ['click']

univariate_dict = {}
for col in cols_to_investigate:
    univariate_dict[col] = {}

firstLine = True
with open(file, 'r') as f:
    # print (f.read())
    for line in f:
        if not firstLine:
            line = line.strip().split(',')
            datetime_data = datetime.strptime(line[cols_to_int_dict['datetime']], "%Y-%m-%d %H:%M:%S")
            
            line.append(datetime_data.year) 
            line.append(datetime_data.month) 
            line.append(datetime_data.day) 
            line.append(datetime_data.hour) 
            line.append(datetime_data.minute) 
            line.append(datetime_data.second) 
            
            for col in cols_to_investigate:
                try:
                    univariate_dict[col][line[cols_to_int_dict[col]]][int(line[cols_to_int_dict['click']])] += 1
                except KeyError:
                    try:
                        univariate_dict[col][line[cols_to_int_dict[col]]][int(line[cols_to_int_dict['click']])] = 1
                    except KeyError:
                        univariate_dict[col][line[cols_to_int_dict[col]]] = {}
                        univariate_dict[col][line[cols_to_int_dict[col]]][int(line[cols_to_int_dict['click']])] = 1
            # print (line)
        else:
            firstLine = False

with open('../inputData/univariate_pickle', 'wb') as f:
    pickle.dump(univariate_dict, f)

print (univariate_dict['browserid'])
print (univariate_dict['countrycode'])
print (univariate_dict['day'])

print (cols_to_int_dict)
