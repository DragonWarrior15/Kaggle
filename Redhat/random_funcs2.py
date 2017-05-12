import pandas as pd
import random
import numpy as np
import datetime
'''
people_dataset = pd.read_csv('originalFiles/people.csv')
people_dataset_2 = people_dataset.rename(index = str, columns = {"char_1" : "pattr_char_1", "group_1" : "pattr_group_1", "char_2" : "pattr_char_2", "date" : "pattr_date", "char_3" : "pattr_char_3", "char_4" : "pattr_char_4", "char_5" : "pattr_char_5", "char_6" : "pattr_char_6", "char_7" : "pattr_char_7", "char_8" : "pattr_char_8", "char_9" : "pattr_char_9", "char_10" : "pattr_char_10", "char_11" : "pattr_char_11", "char_12" : "pattr_char_12", "char_13" : "pattr_char_13", "char_14" : "pattr_char_14", "char_15" : "pattr_char_15", "char_16" : "pattr_char_16", "char_17" : "pattr_char_17", "char_18" : "pattr_char_18", "char_19" : "pattr_char_19", "char_20" : "pattr_char_20", "char_21" : "pattr_char_21", "char_22" : "pattr_char_22", "char_23" : "pattr_char_23", "char_24" : "pattr_char_24", "char_25" : "pattr_char_25", "char_26" : "pattr_char_26", "char_27" : "pattr_char_27", "char_28" : "pattr_char_28", "char_29" : "pattr_char_29", "char_30" : "pattr_char_30", "char_31" : "pattr_char_31", "char_32" : "pattr_char_32", "char_33" : "pattr_char_33", "char_34" : "pattr_char_34", "char_35" : "pattr_char_35", "char_36" : "pattr_char_36", "char_37" : "pattr_char_37", "char_38" : "pattr_char_38"})
people_dataset_2.to_csv('people_dataset.csv', index = False)

activity_dataset = pd.read_csv('originalFiles/act_test.csv')
activity_dataset_2 = activity_dataset.rename(index = str, columns = {"activity_id" : "aattr_activity_id", "date" : "aattr_date", "activity_category" : "aattr_activity_category", "char_1" : "aattr_char_1", "char_2" : "aattr_char_2", "char_3" : "aattr_char_3", "char_4" : "aattr_char_4", "char_5" : "aattr_char_5", "char_6" : "aattr_char_6", "char_7" : "aattr_char_7", "char_8" : "aattr_char_8", "char_9" : "aattr_char_9", "char_10" : "aattr_char_10"})
activity_dataset_2.to_csv('activity_dataset_test.csv', index = False)

#merge datasets on people_id
#people_dataset_2 = pd.read_csv('people_dataset.csv')
#activity_dataset_2 = pd.read_csv('activity_dataset_test.csv')
merged_dataset_test = pd.merge(activity_dataset_2, people_dataset_2, on = 'people_id', how = 'inner')
merged_dataset_test.to_csv('merged_dataset_test.csv', index = False)
'''

'''
print('getting no of rows')
f = open('merged_dataset_test.csv', 'r')
nrows = -1
for line in f:
	nrows = nrows + 1
f.close()
print('no of rows calculated : ' + str(nrows))
'''

'''
f = open('merged_dataset_test_random.csv', 'w')
f.write('merged_dataset_test_random\n')
for i in range(nrows):
	f.write(str(random.uniform(0,1))+'\n')
f.close()
'''

'''
print('preparing sample')
f_w = open('merged_dataset_test_0.1.csv', 'w')
f_r_1 = open('merged_dataset_test.csv', 'r')
f_r_2 = open('merged_dataset_test_random.csv', 'r')

f_w.write(f_r_1.readline())
f_r_2.readline()
for i in range(nrows):
	random_no_char = f_r_2.readline().strip('\n')
	#print(random_no_char, i)
	random_no = float(random_no_char)
	line_to_write = f_r_1.readline()
	if(random_no < 0.1):
		f_w.write(line_to_write)

f_w.close()
f_r_2.close()
f_r_1.close()
print('sample prepared')
'''


def get_woe(cat_col_list, target_col, dataFrame, out_csv):
	#assume the target values are 1's and 0's
	nrows= dataFrame.shape[0]

	#print(dataFrame[cat_col_list[0]].head())

	f = open(out_csv, 'w')
	for col in cat_col_list:
		#get the list of unique values
		col_list = dataFrame[col].values.tolist()
		uniq_val_dict={}
		for i in col_list:
			if i not in uniq_val_dict:
				uniq_val_dict[i] = [0, 0, 0]
		for key in uniq_val_dict:
			events = dataFrame.loc[(dataFrame[col] == key) & (dataFrame[target_col] == 1)].shape[0]
			non_events = dataFrame.loc[(dataFrame[col] == key) & (dataFrame[target_col] == 0)].shape[0]
			woe = np.log(non_events/events) if events != 0 and non_events != 0 else 0
			uniq_val_dict[key] = [events, non_events, woe]
			f.write(col + ',' + str(key) + ',' + str(woe) + '\n')
		
		dataFrame[str('woe_' + str(col))] = dataFrame.apply(lambda row : uniq_val_dict[row[col]][2], axis = 1)
	f.close()
	return(dataFrame)

def set_woe(cat_col_list, dataFrame, in_csv):
	f = open(in_csv, 'r')
	woe_dict = {}
	for line in f:
		temp_var = line.strip('\n').split(',')
		if temp_var[0] not in woe_dict:
			woe_dict[temp_var[0]] = [[], []]
		if temp_var[1] == 'nan': temp_var[1] = np.nan
		woe_dict[temp_var[0]][0].append(temp_var[1])
		woe_dict[temp_var[0]][1].append(float(temp_var[2]))
	f.close()

	for col in cat_col_list:
		woe_dict_temp = {}
		for index, i in enumerate(woe_dict[col][0]):
			woe_dict_temp[i] = woe_dict[col][1][index]
		#print(col)
		#print(woe_dict[col])
		#print(woe_dict_temp)
		dataFrame[str('woe_' + str(col))] = dataFrame.apply(lambda row : woe_dict_temp[row[col]] if row[col] in woe_dict_temp else 0, axis = 1)

	return(dataFrame)


input_file = 'merged_dataset_test_type1.csv'
type1 = True

merged_dataset_test = pd.read_csv(input_file)
#get important info from months
print("Processing dates")
for col in ["pattr_date", "aattr_date"]:
	merged_dataset_test[col + "_month"] = merged_dataset_test.apply(lambda row : datetime.datetime.strptime(str(row[col]), "%Y-%m-%d").date().month, axis = 1)
	merged_dataset_test[col + "_day"] = merged_dataset_test.apply(lambda row : datetime.datetime.strptime(str(row[col]), "%Y-%m-%d").date().day, axis = 1)
	merged_dataset_test[col + "_weekday"] = merged_dataset_test.apply(lambda row : datetime.datetime.strptime(str(row[col]), "%Y-%m-%d").date().weekday(), axis = 1)
print("Dates processed")
#merged_dataset_test = pd.read_csv('merged_dataset_test.csv')

#print(merged_dataset_test.dtypes)

cat_var_list_type1 = ["aattr_date_month","aattr_date_day","aattr_date_weekday","pattr_date_month","pattr_date_day","pattr_date_weekday",
               "aattr_char_1", "aattr_char_2", "aattr_char_3", "aattr_char_4", "aattr_char_5", 
               "aattr_char_6", "aattr_char_7", "aattr_char_8", "aattr_char_9", "pattr_char_1", "pattr_group_1", 
               "pattr_char_2", "pattr_char_3", "pattr_char_4", "pattr_char_5", "pattr_char_6", "pattr_char_7", "pattr_char_8", 
               "pattr_char_9"]


cat_var_list_not_type1 = ["aattr_date_month","aattr_date_day","aattr_date_weekday","pattr_date_month","pattr_date_day","pattr_date_weekday",
               "aattr_activity_category", "aattr_char_10", "pattr_char_1", "pattr_group_1", 
               "pattr_char_2", "pattr_char_3", "pattr_char_4", "pattr_char_5", "pattr_char_6", "pattr_char_7", "pattr_char_8", 
               "pattr_char_9"]


bool_var_list = ["pattr_char_10", "pattr_char_11", "pattr_char_12", "pattr_char_13", "pattr_char_14", "pattr_char_15", 
                "pattr_char_16", "pattr_char_17", "pattr_char_18", "pattr_char_19", "pattr_char_20", "pattr_char_21", 
                "pattr_char_22", "pattr_char_23", "pattr_char_24", "pattr_char_25", "pattr_char_26", "pattr_char_27", 
                "pattr_char_28", "pattr_char_29", "pattr_char_30", "pattr_char_31", "pattr_char_32", "pattr_char_33", 
                "pattr_char_34", "pattr_char_35", "pattr_char_36", "pattr_char_37"]


print("calculating woe's")
if type1:
	merged_dataset_test = set_woe(cat_var_list_type1, merged_dataset_test, 'woe_type1.csv')
else:
	merged_dataset_test = set_woe(cat_var_list_not_type1, merged_dataset_test, 'woe_not_type1.csv')
#merged_dataset_test = set_woe(cat_var_list, merged_dataset_test, 'woe_type1.csv')
print("woe's calculated")


print('converting bools to ind vars')
for col in bool_var_list:
	merged_dataset_test['ind_' + col] = merged_dataset_test.apply(lambda row : 1 if row[col] else 0, axis = 1)
	#merged_dataset_test['ind_' + col] = merged_dataset_test.apply(lambda row : 1 if row[col] else 0, axis = 1)
print('converted bools to ind vars')


#merged_dataset_test['ind_aattr_char_6_type1'] = merged_dataset_test.apply(lambda row : 1 if row['aattr_char_6'] == 'type 1' else 0, axis = 1)
all_col_list = merged_dataset_test.columns.values.tolist()
#all_col_list = merged_dataset_test.columns.values.tolist()
keep_col_list = []
ignore_list_type1=["aattr_activity_category","aattr_char_10"]
ignore_list_not_type1=["aattr_char_1", "aattr_char_2", "aattr_char_3", "aattr_char_4", "aattr_char_5", "aattr_char_6", "aattr_char_7", "aattr_char_8", "aattr_char_9"]
for i in all_col_list:
	if type1:
		if i not in cat_var_list_type1 and i not in ["aattr_date", "people_id", "pattr_date"] and i not in bool_var_list and i not in ignore_list_type1:
			keep_col_list.append(i)
	else:
		if i not in cat_var_list_not_type1 and i not in ["aattr_date", "people_id", "pattr_date"] and i not in bool_var_list and i not in ignore_list_not_type1:
			keep_col_list.append(i)

print('writing modified sampled dataset to csv')
merged_dataset_test[keep_col_list].to_csv(path_or_buf = input_file[:-4] + '_mod.csv', index = False)
#merged_dataset_test[keep_col_list].to_csv(path_or_buf = 'merged_dataset_test_type1_mod.csv', index = False)



#get the correlation matrix
#pd.read_csv('merged_dataset_test_0.1_not_type1_mod.csv').corr().to_csv('corr_test_not_type1.csv')

'''
merged_dataset_test = pd.read_csv('merged_dataset_test_0.1.csv')
merged_dataset_test = merged_dataset_test[merged_dataset_test['aattr_activity_category'] != 'type 1']
merged_dataset_test.to_csv('merged_dataset_test_0.1_not_type1.csv', index = False)

merged_dataset_test = pd.read_csv('merged_dataset_test_0.1.csv')
merged_dataset_test = merged_dataset_test[merged_dataset_test['aattr_activity_category'] == 'type 1']
merged_dataset_test.to_csv('merged_dataset_test_0.1_type1.csv', index = False)
'''