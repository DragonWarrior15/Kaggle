from os import listdir, walk
from os.path import isfile, join
#myPath = r"C:\Users\vaibhav.ojha\Desktop\New folder"
#myPath = r"C:\Users\vaibhav.ojha\Desktop\SH_CVS"
myPath = r"/home/vaibhav.ojha2/kettle/corr_id_matrices"
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#f_w = open('outputFile.txt', 'w')

outputFile = open('corr_top5.txt', 'w')

for dirName, subDirList, fileList in walk(myPath, topdown = True):
  #print(dirName)
  for name in fileList:
    if ('corr' in name.lower()):
      print(join(dirName, name))
      print (name)
      id_key = name.split('_')[1][:-4]
      print id_key

      f = open(join(dirName, name))
      temp_list = []
      for line in f:
        temp_line = line.strip().split(',')
        if 'id' not in temp_line[0] and 'y' not in temp_line[0]:
          try:
            temp_float = abs(float(temp_line[1]))
          except:
            temp_float = 0.0
          temp_list.append([temp_line[0], temp_float])

      f.close()
      temp_list = sorted(temp_list, reverse = True, key = lambda x: x[1])
      outputFile.write(str(id_key))
      for i in range(5):
        outputFile.write(',' + temp_list[i][0] + ',' + str(temp_list[i][1]))
      outputFile.write('\n')

outputFile.close()
