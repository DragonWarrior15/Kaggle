import pandas as pd
import numpy as np
import sys

df_price       = pd.read_csv('../inputs/price_sales_promotion.csv')
df_demo        = pd.read_csv('../inputs/demographics.csv')
df_hist_volume = pd.read_csv('../inputs/historical_volume.csv')
df_weather     = pd.read_csv('../inputs/weather.csv')

df_weather.sort_values(by = ['Agency', 'YearMonth'], inplace = True)
df_price = pd.merge(left = df_price, right = df_hist_volume, on = ['Agency', 'SKU', 'YearMonth'], how = 'inner')
df_price['Total_Sales'] = df_price['Sales'] * df_price['Volume']

## find the places most similar to Agency 06 and 14 in terms of weather
agency_weather = {}
for agency in df_weather['Agency'].unique().tolist():
    agency_weather[agency] = np.array(df_weather.loc[df_weather['Agency'] == agency, 'Avg_Max_Temp'].tolist())

print ('Closest agency through weather')
for agencies in ['Agency_06', 'Agency_14']:
    max_corr = -100
    best_agency = ''

    for agency in agency_weather:
        if agency != agencies:
            corr_coef = np.corrcoef(agency_weather[agencies], agency_weather[agency])[0][1]
            if corr_coef > max_corr:
                max_corr = corr_coef
                best_agency = agency

    print (agencies, best_agency, max_corr)

## find the places most similar to Agency 06 and 14 in terms of population and income
agency_demo = {}
for agency in df_demo['Agency'].unique().tolist():
    agency_demo[agency] = [df_demo.loc[df_demo['Agency'] == agency, 'Avg_Population_2017'].values[0], \
                           df_demo.loc[df_demo['Agency'] == agency, 'Avg_Yearly_Household_Income_2017'].values[0]]

weight_population = 0.2
weight_income = 1 - weight_population

print ()
print ('Closest agency through demographics')
for agencies in ['Agency_06', 'Agency_14']:
    min_distance = 10000000000
    best_agency = ''

    for agency in agency_demo:
        if agency != agencies:
            distance = weight_population * (abs(agency_demo[agency][0] - agency_demo[agencies][0])) + \
                       weight_income * (abs(agency_demo[agency][1] - agency_demo[agencies][1]))
            if distance < min_distance:
                min_distance = distance
                best_agency = agency

    print (agencies, best_agency, min_distance)


print ()
print ('Agency wise Sales of SKUs')
df_price = df_price.loc[df_price['YearMonth'] % 100 == 1, :].groupby(['Agency', 'SKU']).mean()['Total_Sales'].reset_index()
## get the sku wise sales of the following agencies in january months
for agency in ['Agency_05', 'Agency_12', 'Agency_55']:
    print (agency)
    print (df_price.loc[df_price['Agency'] == agency, :].sort_values(by = 'Total_Sales', ascending = False))
    print ()

'''
Closest agency through weather
Agency_06 Agency_05 1.0
Agency_14 Agency_12 1.0

Closest agency through demographics
Agency_06 Agency_55 23879.4
Agency_14 Agency_06 25406.2

Agency wise Sales of SKUs
Agency_05
       Agency     SKU   Total_Sales
26  Agency_05  SKU_01  9.532927e+06
29  Agency_05  SKU_04  6.334662e+06
27  Agency_05  SKU_02  1.020482e+06
28  Agency_05  SKU_03  9.792155e+05
30  Agency_05  SKU_05  4.640513e+05
31  Agency_05  SKU_14  1.176017e+04
32  Agency_05  SKU_21  9.793399e+03
33  Agency_05  SKU_23  5.647134e+02
34  Agency_05  SKU_26  9.335790e+01

Agency_12
       Agency     SKU   Total_Sales
66  Agency_12  SKU_01  1.075444e+07
67  Agency_12  SKU_02  4.161558e+06
70  Agency_12  SKU_05  2.568450e+06
69  Agency_12  SKU_04  2.436373e+06
68  Agency_12  SKU_03  7.300928e+05
71  Agency_12  SKU_07  1.504313e+04

Agency_55
        Agency     SKU   Total_Sales
312  Agency_55  SKU_03  5.558932e+06
310  Agency_55  SKU_01  4.111695e+06
313  Agency_55  SKU_04  3.158137e+06
311  Agency_55  SKU_02  1.237994e+06
314  Agency_55  SKU_05  3.197496e+05
'''


