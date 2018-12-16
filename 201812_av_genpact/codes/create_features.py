import pandas as pd
import numpy as np

def create_features(df, df_center, df_meal, past_week_delta = 1, future_week_delta = 10,
                    week_cutoff = 1, output_path = 'train_file.csv', is_train = True):
    '''
    a function to create input features based on the training dataset
    for this competition, it is assumed that raw datasets are provided
    this dataset will have input as a combination of city, meal and week index (1 to 10)
    output will be the corresponding week indexs' demand, modelling can be done in
    any fashion as suited later on
    df_train - training dataset, contains week X center X meal
    df_meal - contains cuisine infoamtion for meal
    df_center - contains center information
    '''

    print('Input Data Shape : ', df.shape)
    df = df.loc[df['week'] >= week_cutoff - past_week_delta, :].copy()
    df.drop(['id'], axis = 1, inplace = True)
    if('num_orders' not in df.columns):
        df['num_orders'] = 1
    # df.fillna({'num_orders':1}, inplace = True)
    # train the model on the log of targets as metric is rmsle
    df['num_orders'] = np.log(df['num_orders'])

    df = pd.merge(left = df, right = df_center, how = 'left', on = 'center_id')
    df = pd.merge(left = df, right = df_meal, how = 'left', on = 'meal_id')

    # one hot encoding of cuisine, center type etc
    for center_type in ['A', 'B']:
        df['center_type_' + center_type] = 0
        df.loc[df['center_type'] == 'TYPE_' + center_type, 'center_type_' + center_type] = 1

    for cuisine in ['Thai', 'Indian', 'Italian']:
        df['cuisine_' + cuisine] = 0
        df.loc[df['cuisine'] == cuisine, 'cuisine_' + cuisine] = 1

    for region in [56, 34, 77, 85, 23, 71, 35, 93]:
        df['region_code_' + str(region)] = 0
        df.loc[df['region_code'] == region, 'region_code_' + str(region)] = 1

    for food_category in ['Beverages', 'Extras', 'Soup', 'Other Snacks', 'Salad', 'Rice Bowl', 'Starters', \
                          'Sandwich', 'Pasta', 'Desert', 'Biryani', 'Pizza', 'Fish']:
        df['food_category_' + food_category.replace(' ', '_')] = 0
        df.loc[df['category'] == food_category, 'food_category_' + food_category.replace(' ', '_')] = 1

    df.drop(['center_type', 'cuisine', 'region_code', 'category'], axis = 1, inplace = True)
    
    min_week = max(week_cutoff - past_week_delta, 1)
    max_week = max(df['week'])
    if(max_week < 0):
        print('max week', max(df['week']), 'future week delta', future_week_delta, 
              'are wrong arguments')
        return(False)
    if(max_week < min_week):
        print('max week', max(df['week']), 'future week delta', future_week_delta, 
              'week cutoff', week_cutoff, 'are wrong arguments')
        return(False)
    
    # additional features on raw dataframe
    df['discount'] = df['base_price'] - df['checkout_price']
    df['discount_pct'] = df['discount']/df['base_price']

    print('Adding previous week features')
    # start preparing week level features
    df_curr_week = df.copy()
    for i in range(1, past_week_delta + 1):
        print('Currently working %d weeks back' % (i))
        df_prev_week = df.copy()
        df_prev_week['week'] = df_prev_week['week'] + i
        df_curr_week = pd.merge(left = df_curr_week, right = df_prev_week, 
                                how = 'left', on = ['week', 'center_id', 'meal_id'],
                                suffixes = ('', '_prev_week_' + str(i)))


        df_curr_week.drop([x for x in df_curr_week.columns.tolist() \
                          if ('center_type'   in x or \
                              'cuisine'       in x or \
                              'region_code'   in x or \
                              'food_category' in x or \
                              'op_area'       in x or \
                              'city_code'     in x) and '_prev_week_' in x],
                          axis = 1, inplace = True)

        # df_curr_week = df_curr_week.loc[~df_curr_week['num_orders_prev_week_' + str(i)].isnull(), :]

    df_curr_week_list = []
    # now add target information by duplicating
    print('Adding next week features')
    for i in range(1, future_week_delta + 1):
        print('Currently working %d weeks forward' % (i))
        df_next_week = df.copy()
        df_next_week['week'] = df_next_week['week'] - i
        df_curr_week_temp = pd.merge(left = df_curr_week, right = df_next_week, 
                                     how = 'left', on = ['week', 'center_id', 'meal_id'],
                                     suffixes = ('', '_target_week'))
        df_curr_week_temp['target_week'] = i
        df_curr_week_temp.rename(columns = {'num_orders_target_week' : 'target'}, inplace = True)

        df_curr_week_temp.drop([x for x in df_curr_week_temp.columns.tolist() \
                               if ('center_type'  in x or \
                                  'cuisine'       in x or \
                                  'region_code'   in x or \
                                  'food_category' in x or \
                                  'op_area'       in x or \
                                  'city_code'     in x) and '_target_week' in x],
                               axis = 1, inplace = True)

        # add features for future information, but j weeks before the target week
        for j in range(1, 3):
            df_next_week = df.copy()
            df_next_week['week'] = df_next_week['week'] - i - j
            df_next_week.drop(['num_orders'], axis = 1, inplace = True)
            df_curr_week_temp = pd.merge(left = df_curr_week_temp, right = df_next_week, 
                                         how = 'left', on = ['week', 'center_id', 'meal_id'],
                                         suffixes = ('', '_target_week_' + str(j) + '_week_before'))

        df_curr_week_temp.drop([x for x in df_curr_week_temp.columns.tolist() \
                              if ('center_type'   in x or \
                                  'cuisine'       in x or \
                                  'region_code'   in x or \
                                  'food_category' in x or \
                                  'op_area'       in x or \
                                  'city_code'     in x) and '_target_week_' in x],
                               axis = 1, inplace = True)

        if(is_train):
            df_curr_week_temp = df_curr_week_temp.loc[~df_curr_week_temp['target'].isnull()]
            df_curr_week_temp.to_csv(output_path[:-4] + '_target_' + str(i) + '_train.csv', index = False)
        else:
            df_curr_week_temp.to_csv(output_path[:-4] + '_target_' + str(i) + '_test.csv', index = False)
        # df_curr_week_list.append(df_curr_week_temp.copy())

    print('Data Prepared !')

    # df_curr_week = pd.concat(df_curr_week_list)
    # print('Data Prepared, Shape : ' + str(df_curr_week.shape))
    # df_curr_week.to_csv(output_path, index = False)
    return(True)


if(__name__ == '__main__'):
    root_path = '../inputs/'

    df = pd.read_csv(root_path + 'train.csv')

    df_test = pd.read_csv(root_path + 'test.csv')
    df_test['num_orders'] = np.nan

    # some handling for the case where a particular center meal combination
    # is absent in the week 145
    df_test_all_comb = df_test.groupby(['center_id', 'meal_id']).count()['id'].reset_index()[['center_id', 'meal_id']]
    df_test_all_comb['week'] = 145
            
    df = pd.concat([df, df_test, df_test_all_comb])
    df = df.drop_duplicates(subset = ['week', 'center_id', 'meal_id'])


    df_center = pd.read_csv(root_path + 'fulfilment_center_info.csv')
    df_meal = pd.read_csv(root_path + 'meal_info.csv')

    past_week_delta = 3
    future_week_delta = 10
    week_cutoff = 1

    output_path = root_path + 'train_past_%d_future_%d_cutoff_%d.csv' % (past_week_delta, future_week_delta, week_cutoff)

    # train
    # create_features(df, df_center, df_meal, 
                    # past_week_delta = past_week_delta, future_week_delta = future_week_delta, 
                    # week_cutoff = week_cutoff, output_path = output_path, is_train = True)

    # test
    create_features(df, df_center, df_meal, 
                    past_week_delta = past_week_delta, future_week_delta = future_week_delta, 
                    week_cutoff = 145, output_path = output_path, is_train = False)


        




    




