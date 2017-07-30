param_dict = {'learning_rate' : [0.05, 1, 3],
              'max_depth' : [5, 10, 50, 100, 200],
              'n_estimators' : [50, 100, 200]}
# param_dict = {'learning_rate' : [0.05, 1, 2, 3],
#               'max_depth' : [5, 10, 30, 50],
#               'n_estimators' : [50, 100, 200, 300, 500]}

param_space = []
param_list = sorted(list([k for k in param_dict]))
for param in param_list:
    curr_param_space_length = len(param_space)
    if (curr_param_space_length == 0):
        for i in range(len(param_dict[param])):
            param_space.append([param_dict[param][i]])
    else:
        for i in range(len(param_dict[param]) - 1):
            for j in range(curr_param_space_length):
                param_space.append(list(param_space[j]) + [param_dict[param][i]])

        for i in range(curr_param_space_length):
            param_space[i].append(param_dict[param][-1])

print (param_space)
print (sorted(param_space))
print (param_space)

for k in sorted(param_space):
    print (k)

print (len(param_space))