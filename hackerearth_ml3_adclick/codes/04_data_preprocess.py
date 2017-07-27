# one hot encode : browserid, countrycode, day, devid
# woe replacement : category, merchant, offerid, siteid
from datetime import datetime

day_to_weekday_dict = {10:'tue',
                       11:'wed',
                       12:'thu',
                       13:'fri',
                       14:'sat',
                       15:'sun',
                       16:'mon',
                       17:'tue',
                       18:'wed',
                       19:'thu',
                       20:'fri',
                       21:'sat',
                       22:'sun',
                       23:'mon',
                       }

# get the woe values
woe_dict = {}
with open('../analysis_graphs/woe.csv', 'r') as f:
    firstLine = True
    # print (f.read())
    for line in f:
        line = line.strip().split(',')
        if not firstLine:
            if line[0] in woe_dict:
                pass
            else:
                woe_dict[line[0]] = {}
            woe_dict[line[0]][line[1]] = float(line[4])
        else:
            firstLine = False

header_list = ['category', 'merchant', 'offerid', 'siteid',
           'browserid_chrome', 'browserid_ie', 'browserid_firefox', 'browserid_opera', 'browserid_safari', 'browserid_blank',
           'countrycode_a', 'countrycode_b', 'countrycode_c', 'countrycode_d', 'countrycode_e', 'countrycode_f',
           'day_sun', 'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri', 'day_sat',
           'devid_desktop', 'devid_mobile', 'devid_tablet', 'devid_blank',
           'ID', 'click']

cols_to_int_dict = {'ID':0,
                    'datetime':1,
                    'siteid':2,
                    'offerid':3,
                    'category':4,
                    'merchant':5,
                    'countrycode':6,
                    'browserid':7,
                    'devid':8,
                    'click':9
                    }

devid_dict = {'Desktop' : [1,0,0,0], 
              'Mobile' : [0,1,0,0], 
              'Tablet' : [0,0,1,0], 
              '' : [0,0,0,1]
              }

browserid_dict = {'Chrome' : [1,0,0,0,0,0],
                  'Google Chrome' : [1,0,0,0,0,0],
                  'InternetExplorer' : [0,1,0,0,0,0],
                  'Edge' : [0,1,0,0,0,0],
                  'Internet Explorer' : [0,1,0,0,0,0],
                  'IE' : [0,1,0,0,0,0],
                  'Mozilla Firefox' : [0,0,1,0,0,0],
                  'Firefox' : [0,0,1,0,0,0],
                  'Mozilla' : [0,0,1,0,0,0],
                  'Opera' : [0,0,0,1,0,0],
                  'Safari' : [0,0,0,0,1,0],
                  '' : [0,0,0,0,0,1]
                  }

countrycode_dict = {'a' : [1,0,0,0,0,0],
                'b' : [0,1,0,0,0,0],
                'c' : [0,0,1,0,0,0],
                'd' : [0,0,0,1,0,0],
                'e' : [0,0,0,0,1,0],
                'f' : [0,0,0,0,0,1]
                }

day_dict = {'sun' : [1,0,0,0,0,0,0],
            'mon' : [0,1,0,0,0,0,0],
            'tue' : [0,0,1,0,0,0,0],
            'wed' : [0,0,0,1,0,0,0],
            'thu' : [0,0,0,0,1,0,0],
            'fri' : [0,0,0,0,0,1,0],
            'sat' : [0,0,0,0,0,0,1]
}

file_list = ['../inputData/train_10to17_train.csv',
             '../inputData/train_18to19_test.csv',
             '../inputData/train_20_val.csv',
             '../inputData/test.csv'
             ]

output_file_list = [open('../inputData/train_10to17_train_processed.csv', 'w'),
                    open('../inputData/train_18to19_test_processed.csv', 'w'),
                    open('../inputData/train_20_val_processed.csv', 'w'),
                    open('../inputData/test_processed.csv', 'w')
                    ]

# file_list = ['../inputData/train_sample.csv']
# output_file_list = [open('../inputData/train_sample_processed.csv', 'w')]


for file, output_file in zip(file_list, output_file_list):
    with open(file, 'r') as f:
        if 'test' in file:
            headers = list(header_list[:-1])
        else:
            headers = list(header_list)
        firstLine = True
        # print (f.read())
        for line in f:
            line = line.strip().split(',')
            str_to_write = []
            if not firstLine:
                datetime_data = datetime.strptime(line[cols_to_int_dict['datetime']], "%Y-%m-%d %H:%M:%S")

                
                for field_name in ['category', 'merchant', 'offerid', 'siteid']:
                    try:
                        str_to_write.append(str(woe_dict[field_name][line[cols_to_int_dict[field_name]]]))
                    except KeyError:
                        str_to_write.append(str(woe_dict[field_name]['OTHERS']))

                for i in browserid_dict[line[cols_to_int_dict['browserid']]]:
                    str_to_write.append(str(i))

                for i in countrycode_dict[line[cols_to_int_dict['countrycode']]]:
                    str_to_write.append(str(i))

                for i in day_dict[day_to_weekday_dict[datetime_data.day]]:
                    str_to_write.append(str(i))

                for i in devid_dict[line[cols_to_int_dict['devid']]]:
                    str_to_write.append(str(i))

                for field_name in ['ID']:
                    str_to_write.append(str(line[cols_to_int_dict[field_name]]))

                if 'test' not in file:
                    for field_name in ['ID']:
                        str_to_write.append(str(line[cols_to_int_dict[field_name]]))

                output_file.write(','.join(str_to_write) + '\n')

            else:
                firstLine = False
                output_file.write(','.join(headers) + '\n')

for file in output_file_list:
    file.close()