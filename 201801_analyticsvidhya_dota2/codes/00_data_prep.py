import pandas as pd
import numpy as np
import os
import sys
import pickle

# read the dataset for the heroes for preprocessing it
df_heroes = pd.read_csv('../inputs/hero_data.csv')

# get the count of unique items in the different columns of the data
print ("%22s, %5s, %4s, %8s" %('Column', 'Uniq', 'nan', 'dtype'))
for col in df_heroes.columns:
    # pass
    print ("%22s, %5d, %4d, %8s" %(col, len(df_heroes[col].unique()), df_heroes[col].isnull().sum(), df_heroes[col].dtype))

                # Column,  Uniq,  nan
               # hero_id,   115,    0
          # primary_attr,     3,    0
           # attack_type,     2,    0
                 # roles,    98,    0
           # base_health,     1,    0
     # base_health_regen,     9,    0
             # base_mana,     1,    0
       # base_mana_regen,     1,    0
            # base_armor,    11,    0
 # base_magic_resistance,     2,    0
       # base_attack_min,    35,    0
       # base_attack_max,    35,    0
         # base_strength,    15,    0
          # base_agility,    20,    0
     # base_intelligence,    18,    0
         # strength_gain,    25,    0
          # agility_gain,    31,    0
     # intelligence_gain,    32,    0
          # attack_range,    24,    0
      # projectile_speed,    13,    0
           # attack_rate,    10,    0
            # move_speed,    15,    0
             # turn_rate,     7,    0

# remove the columns with a single unique value
df_heroes = df_heroes[[x for x in df_heroes.columns.tolist() if len(df_heroes[x].unique()) > 1]]

## start processing the string columns to convert them to one hot encoded
## versions
# study the role variables, it is comprised of multiple strings joined together
# let's get a list of all unique strings in that column
roles_list = sorted(list(set(':'.join(df_heroes['roles'].tolist()).split(':'))), key = lambda x: x)
print (roles_list)
# ['Carry', 'Disabler', 'Durable', 'Escape', 'Initiator', 'Jungler', 'Nuker', 'Pusher', 'Support']

# Create new columns indicating whether the hero can play the above derived roles or not
for role in roles_list:
    df_heroes[('Role_' + role).lower()] = df_heroes['roles'].apply(lambda x: 1 if role in set(x.split(':')) else 0)
df_heroes.drop(['roles'], axis = 1, inplace = True)


# process the primary attr columns
primary_attr_list = sorted(df_heroes['primary_attr'].unique().tolist())
for attr in primary_attr_list:
    df_heroes[('primary_attr_' + attr).lower()] = df_heroes['primary_attr'].apply(lambda x: 1 if x == attr else 0)
df_heroes.drop(['primary_attr'], axis = 1, inplace = True)


# process the attack type columns
attack_type_list = sorted(df_heroes['attack_type'].unique().tolist())
for attack in attack_type_list:
    df_heroes[('attack_type_' + attack).lower()] = df_heroes['attack_type'].apply(lambda x: 1 if x == attack else 0)
df_heroes.drop(['attack_type'], axis = 1, inplace = True)

print (df_heroes.columns.tolist())
# ['hero_id', 'base_health_regen', 'base_armor', 'base_magic_resistance', 'base_attack_min', 
 # 'base_attack_max', 'base_strength', 'base_agility', 'base_intelligence', 'strength_gain', 
 # 'agility_gain', 'intelligence_gain', 'attack_range', 'projectile_speed', 'attack_rate', 
 # 'move_speed', 'turn_rate', 'role_carry', 'role_disabler', 'role_durable', 'role_escape', 
 # 'role_initiator', 'role_jungler', 'role_nuker', 'role_pusher', 'role_support', 
 # 'primary_attr_agi', 'primary_attr_int', 'primary_attr_str', 'attack_type_melee', 
 # 'attack_type_ranged']

# in the hero data, the hero_id are not contiguous, some are missing in between
# create a dictionary that maps the actual hero id to the row number
# for further use in train datasets
hero_id_dict = dict([[x, index] for index, x in enumerate(df_heroes['hero_id'].tolist())])
with open('../inputs/hero_id_dict', 'wb') as f:
    pickle.dump(hero_id_dict, f)

df_heroes.to_csv('../inputs/hero_data_processed.csv', index = False)
