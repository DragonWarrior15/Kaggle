import random

inputFileName = 'data.csv'
outputFileName_train = inputFileName[:-4] + '_train.csv'
outputFileName_test = inputFileName[:-4] + '_test.csv'
headers = True

f_w_train = open(outputFileName_train, 'w')
f_w_test = open(outputFileName_test, 'w')

p = 0.3

f = open(inputFileName, 'r')
nrows = 0
for line in f:
  tempVar = line
  if headers and nrows == 0:
    f_w_train.write(tempVar)
    f_w_test.write(tempVar)
  else:
    if random.random() > 0.3:
      f_w_train.write(tempVar)
    else:
      f_w_test.write(tempVar)
  nrows += 1

f.close()
f_w_train.close()
f_w_test.close()
