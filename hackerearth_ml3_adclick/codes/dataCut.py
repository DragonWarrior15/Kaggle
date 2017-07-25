dataDir = 'inputs'
inputFileName = 'act_test.csv'
outputFileName = inputFileName[:-4] + '_part.csv'
headers = True

f = open(dataDir + '/' + inputFileName, 'r')
# get the columns
colList = f.readline().strip('\n').split(',')
#print (colList)
f.close()

colDict = {}
for i in range(len(colList)):
	colDict[colList[i]] = i

# colList -> ['people_id', 'activity_id', 'date', 'activity_category', 'char_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6',
#             'char_7', 'char_8', 'char_9', 'char_10', 'outcome']

# colList -> ['people_id', 'char_1', 'group_1', 'char_2', 'date', 'char_3', 'char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9', 
#             'char_10', 'char_11', 'char_12', 'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20', 
#             'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30', 'char_31', 
#             'char_32', 'char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38']


#keepList = ['people_id', 'date', 'outcome']
keepList = ['people_id', 'date']
#keepList = ['people_id', 'char_2', 'date', 'char_38', 'group_1']

nrows = -1 if headers else 0
f_w = open(dataDir + '/' + outputFileName, 'w')


with open(dataDir + '/' + inputFileName, 'r') as f:
	for line in f:
		nrows = nrows + 1
		temp_var = line.strip('\n').split(',')
		line_to_write = ''
		for i in keepList:
			line_to_write = line_to_write + temp_var[colDict[i]] + ','
		line_to_write = line_to_write.strip(',') + '\n'

		f_w.write(line_to_write)

f_w.close()
