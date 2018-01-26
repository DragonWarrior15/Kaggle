Dataset Descriptions

train9.csv and train1.csv contain the user performance for their most frequent 9 heroes and 10th hero respectively. Both train9.csv and train1.csv have below fields.

Feature              Description
user_id              The id of the user
hero_id              The id of the hero the player played with
id                   Unique id
num_games            The number of games the player played with that hero
num_wins             Number of games the player won with this particular hero
kda_ratio (target)   ((Kills + Assists)*1000/Deaths) 

Ratio: where kill, assists and deaths are average values per match for that hero

test9.csv contain the different set of user (different from training set) performance for their most frequent 9 heroes. test9.csv has similar fields as train9.csv. Now, the aim is to predict the performance (kda_ratio) of the same set of users (test users) with the 10th hero which is test1.csv.

Feature        Description
user_id        The id of the user
hero_id        The id of the hero (of which the kda_ratio has to be predicted)
id             Unique id
num_games      The total number of games the player played with this hero

We also have "hero_data.csv" which contains information about heros.

Feature         Description
hero_id         Id of the hero
primary_attr    A string denoting what the primary attribute of the hero is
                (int- initiator, agi- agility, str- strength and so on)
attack_type     String, :"Melee" or "Ranged"
roles           An array of strings which have roles of heroes
                (eg Support, Disabler, Nuker, etc.)
base_health     The basic health the hero starts with
base_health_regen,base_mana,base_mana_regen,
base_armor,base_magic_restistance,
base_attack_min,base_attack_max,base_strength,
base_agility,base_intelligence,strength_gain,agility_gain,intelligence_gain,
attack_range,projectile_speed,attack_rate,move_speed,turn_rate
These are the basic stats the heroes start with 
(some remain same throughout the game)

Evaluation Metric
The predictions will be evaluated on RMSE.

The public private split is 40:60