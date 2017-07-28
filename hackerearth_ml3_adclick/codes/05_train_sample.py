import numpy as np

np.random.seed(42)

file = '../inputData/train_10to18_train_processed.csv'
output_file = '../inputData/train_10to18_train_processed_sample.csv'

output_file = open(output_file, 'w')

pct_data = 0.4

with open(file, 'r') as f:
    firstLine = True
    for line in f:
        if not firstLine:
            if np.random.rand() < pct_data:
                output_file.write(line + '\n')
        else:
            firstLine = False
            output_file.write(line + '\n')

output_file.close()