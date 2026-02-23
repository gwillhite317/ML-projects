# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


data_og = pd.read_csv("C:/Users/mowma/Downloads/NFL Play by Play 2009-2018 (v5).csv/NFL Play by Play 2009-2018 (v5).csv")
print(data_og[['down', 'play_type', 'game_seconds_remaining', 'home_team', 'ydstogo']].head(10))

# %%
valid_plays = ['run','pass','punt','field_goal' ]
data_4th = data_og[(data_og['down'] == 4) &
                   (data_og['play_type'].isin(valid_plays))
                   ].copy()

data_4th = data_4th[data_4th['penalty'] == 0]
data_4th['u2m'] = (data_4th['game_seconds_remaining'] < 120).astype(int)
data_4th['u1m_half'] = (data_4th['half_seconds_remaining'] < 120).astype(int)
data_4th['is_home'] = (data_4th['posteam'] == data_4th['home_team']).astype(int)

# %%
data_4th['ytg_buck'] = pd.cut(data_4th['ydstogo'],
                              bins = [0, 3, 6, 10, 15, 100],
                              labels=['0-3','3-6','7-10','11-15','15+'])

data_4th['sdiff_buck'] = pd.cut(data_4th['score_differential'],
                                bins = [-100, -24, -17, -10, -7, -3, 3, 7, 10, 17, 24, 100],
                                labels=['neg_big', 'neg_4','neg_3','neg_2', 'neg_1','close','pos_1','pos_2','pos_3','pos_4','big_pos']
                                )

d_4_r = data_4th.sort_values(['season', 'gameid','playtimediff'])