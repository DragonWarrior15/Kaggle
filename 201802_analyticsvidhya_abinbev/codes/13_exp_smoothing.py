import pandas as pd
import numpy as np
import sys
from scipy.optimize import differential_evolution
import pickle

def competition_metric(y_true, y_forecast):
    y_true = np.array(y_true)
    y_forecast = np.array(y_forecast)
    return (1 - (np.sum(np.abs(y_true - y_forecast)) / np.sum(y_true)))

def competition_metric_minimize(y_true, y_forecast):
    y_true = np.array(y_true)
    y_forecast = np.array(y_forecast)
    return (np.sum(np.abs(y_true - y_forecast)) / np.sum(y_true))

def competition_scorer(estimator, X, y_true):
    return(1 - (np.sum(np.abs(y_true - estimator.predict(X))) / np.sum(y_true)))


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        try:
            sum += float(series[i+slen] - series[i]) / slen
        except IndexError:
            break
    return sum / slen

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result

df_hist_volume = pd.read_csv('../inputs/historical_volume.csv')
df_hist_volume.sort_values(by = ['Agency', 'SKU', 'YearMonth'], inplace = True)
df_hist_volume = df_hist_volume.loc[df_hist_volume['Volume'] > 0, :]


try:
    with open('agency_sku_dict', 'rb') as f:
        agency_sku_dict = pickle.load(f)
except FileNotFoundError:

    ## prepare the inputs
    print ('Preparing Inputs')
    agency_sku_dict = {}
    for group_obj in df_hist_volume.groupby(['Agency', 'SKU']):
        agency = group_obj[0][0]
        sku = group_obj[0][1]
        print ('Current Agency : ' + agency + ' Current SKU : ' + sku)
        try:
            agency_sku_dict[agency][sku] = {}
        except KeyError:    
            agency_sku_dict[agency] = {}
            agency_sku_dict[agency][sku] = {}

        for month in df_hist_volume.loc[(df_hist_volume['Agency'] == agency) & (df_hist_volume['SKU'] == sku), 'YearMonth'].tolist() + [201801]:
            if month >= 201501:
                ts = df_hist_volume.loc[(df_hist_volume['Agency'] == agency) & (df_hist_volume['SKU'] == sku) & (df_hist_volume['YearMonth'] < month), 'Volume'].tolist()
                try:
                    target = df_hist_volume.loc[(df_hist_volume['Agency'] == agency) & (df_hist_volume['SKU'] == sku) & (df_hist_volume['YearMonth'] == month), 'Volume'].values[0]
                except IndexError:
                    target = 0
                if len(ts) >= 13:
                    agency_sku_dict[agency][sku][month] = []
                    agency_sku_dict[agency][sku][month].append(ts)
                    agency_sku_dict[agency][sku][month].append(target)
    print ('Inputs Prepared')

    with open('agency_sku_dict', 'wb') as f:
        pickle.dump(agency_sku_dict, f)    


total_params = 0
for agency in agency_sku_dict:
    # for sku in agency_sku_dict[agency]:
    total_params += 1
total_params *= 3
total_params = 3
global_params_bounds = [(0, 1) for i in range(total_params)]

def optimization_function(params_list, agency_sku_dict):
    if isinstance(agency_sku_dict, list):
        agency_sku_dict = agency_sku_dict[0]

    y_true = []
    y_pred = []
    num_skus = []
    for agency_index, agency in enumerate(sorted(agency_sku_dict.keys())):
        num_skus.append(1)
        for sku_index, sku in enumerate(sorted(agency_sku_dict[agency].keys())):
            num_skus[-1] += 1
            
            # alpha, beta, gamma = params_list[sum(num_skus[:-1]) + sku_index : sum(num_skus[:-1]) + sku_index + 3]
            alpha, beta, gamma = params_list
            # params = agency_sku_dict[agency][sku]['params']
            # alpha, beta, gamma = params[0], params[1], params[2]

            for month in agency_sku_dict[agency][sku]:
                if month != 201801:
                    pred = triple_exponential_smoothing(agency_sku_dict[agency][sku][month][0], 12, alpha, beta, gamma, 1)
                    y_pred.append(pred[-1])
                    y_true.append(agency_sku_dict[agency][sku][month][1])

    metric = competition_metric_minimize(y_true, y_pred)
    return (metric)

print ('Training Evolutionary Algorithm')
result = differential_evolution(func = optimization_function, bounds = global_params_bounds, maxiter = 100,\
                                popsize = 10, args = [agency_sku_dict], seed = 42)
print (result.x)
print (result.fun)

best_x = result.x

# [0.47855363 0.00539747 0.        ]  0.14239818371010837, 0.16784176258917302

# test on january 2017
agency_sku_dict_201701 = {}
for agency in agency_sku_dict:
    agency_sku_dict_201701[agency] = {}
    for sku in agency_sku_dict[agency]:
        agency_sku_dict_201701[agency][sku] = {}
        try:
            agency_sku_dict_201701[agency][sku][201701] = agency_sku_dict[agency][sku][201701]
        except KeyError:
            pass

print ('Metric on 201701 : ' + str(optimization_function(best_x, agency_sku_dict_201701)))


def make_predictions(row):
    alpha, beta, gamma = best_x
    try:
        pred = triple_exponential_smoothing(agency_sku_dict[row['Agency']][row['SKU']][201801][0], 12, alpha, beta, gamma, 1)[-1]
        pred = 0 if pred < 0 else pred
    except KeyError:
        pred = 0
    return pred

## prepare submissions
submission_file = pd.read_csv('../inputs/volume_forecast.csv')
submission_file['Volume'] = submission_file.apply(lambda row: make_predictions(row), axis = 1)
submission_file.fillna(0, inplace = True)
submission_file.to_csv('../submissions/submit_exp_20180225_1630.csv', index = False)

