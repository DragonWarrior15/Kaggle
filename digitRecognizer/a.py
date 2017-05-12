f = open('digitRecognizerOutput003.csv', 'r')
f_w = open('out.csv', 'w')
firstLine = True
for line in f:
	if firstLine:
		f_w.write(line)
		firstLine = False
	else:
		temp_var = line.strip('\n').split(',')
		temp_var = [int(num) for num in temp_var]
		temp_var[0] = temp_var[0] + 1

		f_w.write(str(temp_var[0])+','+str(temp_var[1])+'\n')

f.close()
f_w.close()