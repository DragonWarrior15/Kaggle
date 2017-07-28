param_dict = {'learning_rate' : [0.05, 1],
              'max_depth' : [5, 10],
              'n_estimators' : [50, 100]}
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
                param_space.append(list(param_space[j]))

        for i in range(len(param_space)):
            param_space[i].append(param_dict[param][i%len(param_dict[param])])

print (param_space)