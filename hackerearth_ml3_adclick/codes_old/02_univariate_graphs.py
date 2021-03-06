from datetime import datetime
import pickle
import numpy as np

columns = ['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click','year','month','day','hour','minute','second']
cols_to_int_dict = dict([[columns[i], i] for i in range(len(columns))])
cols_to_investigate = ['siteid','category','merchant','countrycode','browserid','devid','day','hour']
target = ['click']

with open('../analysis_graphs/univariate_pickle', 'rb') as f:
    univariate_dict = pickle.load(f)

# print ([k for k in univariate_dict])

for k in univariate_dict:
    univariate_dict[k]['OTHERS'] = {0:0,1:0}
    key_to_del = []
    for k1 in univariate_dict[k]:
        if 0 not in univariate_dict[k][k1] or 1 not in univariate_dict[k][k1]:
            for i in [0, 1]:
                try:
                    univariate_dict[k]['OTHERS'][i] += univariate_dict[k][k1][i]
                except KeyError:
                    pass
            key_to_del.append(k1)

    for k1 in key_to_del:
        del univariate_dict[k][k1]

    if univariate_dict[k]['OTHERS'][0] == 0 and univariate_dict[k]['OTHERS'][1] == 0:
        del univariate_dict[k]['OTHERS']

# code to map some of the values
for k in ['Google Chrome']:
    for i in [0, 1]:
        univariate_dict['browserid']['Chrome'][i] += univariate_dict['browserid'][k][i]
    del univariate_dict['browserid'][k]

for k in ['IE', 'Internet Explorer', 'Edge']:
    for i in [0, 1]:
        univariate_dict['browserid']['InternetExplorer'][i] += univariate_dict['browserid'][k][i]
    del univariate_dict['browserid'][k]

for k in ['Mozilla Firefox', 'Mozilla']:
    for i in [0, 1]:
        univariate_dict['browserid']['Firefox'][i] += univariate_dict['browserid'][k][i]
    del univariate_dict['browserid'][k]

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

univariate_dict['day_temp'] = univariate_dict['day']
del univariate_dict['day']
univariate_dict['day'] = {}
for x in ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']:
    univariate_dict['day'][x] = {0:0,1:1}
    for k in univariate_dict['day_temp']:
        if x == day_to_weekday_dict[k]:
            for i in [0, 1]:
                univariate_dict['day'][x][i] += univariate_dict['day_temp'][k][i]
del univariate_dict['day_temp']

with open('../analysis_graphs/univariate_profile.csv', 'w') as f:
    f.write('variable,variable_value,0,1\n')
    for k in univariate_dict:
        for k1 in univariate_dict[k]:
            value_0 = str(univariate_dict[k][k1][0])
            value_1 = str(univariate_dict[k][k1][1])
            f.write(str(k) + ',' + str(k1) + ',' + value_0 + ',' + value_1 + '\n')

# print (univariate_dict['browserid'])
# print (univariate_dict['countrycode'])
# print (univariate_dict['day'])

# print (cols_to_int_dict)

# calculation of woe
# woe = ln(%non events / %events)
# %events = events in bin/total events in the variable
# %non events = non events in bin/total non events in the variable

cols_to_investigate = ['siteid', 'offerid', 'category','merchant','countrycode','browserid','devid','day','hour']
with open('../analysis_graphs/woe.csv', 'w') as f:
    f.write('variable,variable_value,pct_0,pct_1,woe\n')
    for dimension in cols_to_investigate:
        total = [1, 1]
        print (dimension)
        total[0] = sum([univariate_dict[dimension][k][0] for k in univariate_dict[dimension]])
        total[1] = sum([univariate_dict[dimension][k][1] for k in univariate_dict[dimension]])
        for dimension_value in univariate_dict[dimension]:
            str_to_write = str(dimension) + ',' + str(dimension_value) + ','
            pct = [0,0]
            for i in [0, 1]:
                if total[i] != 0:
                    pct[i] = univariate_dict[dimension][dimension_value][i]/(1.0 * total[i])
                    str_to_write += str(pct[i]) + ','
                else:
                    str_to_write += ','

            if pct[0] == 0 or pct[1] == 0:
                str_to_write += ','
            else:
                str_to_write += str(np.log(pct[0]/pct[1])) + ','

            str_to_write += '\n'
            f.write(str_to_write)