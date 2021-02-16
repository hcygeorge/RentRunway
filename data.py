#%%
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import json
import pandas as pd
from scipy import sparse
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

#%% Load json data
with open('./data/renttherunway_final_data.json') as f:
    raw_data = []
    for line in f.readlines():
        record = json.loads(line)
        raw_data.append(record)

raw_df = pd.DataFrame(raw_data)

#%% Data description
raw_df.shape  # (192544, 15)

raw_df.columns = ['fit', 'user_id', 'bust_size', 'item_id', 'weight',
                    'rating', 'rented_for', 'review_text', 'body_type',
                    'review_summary', 'category', 'height', 'size1',
                    'age', 'review_date']  # 15 columns

#%% Drop features and rating missed rows
raw_df.rating = raw_df.rating.astype(float) / 2
raw_df = raw_df[raw_df.rating.notnull()]  # 82 missing ratings

#%% Count sparse rate
def sparsity(df):
    unique_user = df.user_id.value_counts()
    unique_item = df.item_id.value_counts()
    num_user = len(unique_user)
    num_item = len(unique_item)
    sparse_rate = (1- (df.shape[0] / (num_user * num_item))) * 100
    print("Unique users/items: {}/{}".format(num_user, num_item))
    print("Matrix sparse rate: {:.2f} %".format(sparse_rate))
    return unique_user, unique_item
    
unique_user, unique_item = sparsity(raw_df)

user_drop = unique_user[unique_user < 3].index  # 71824 uncommon users
item_drop = unique_item[unique_item < 3].index  # 341 uncommon items

rating_df1 = raw_df.set_index('user_id', inplace=False)
rating_df2 = rating_df1.drop(user_drop, axis=0)

unique_item2 = rating_df2.item_id.value_counts()
item_drop2 = unique_item2[unique_item2 < 3].index  # 417

rating_df2.reset_index(level='user_id', inplace=True)
rating_df2.set_index('item_id', inplace=True)
rating_df3 = rating_df2.drop(item_drop2, axis=0)
rating_df3.reset_index(level='item_id', inplace=True)

_, _ = sparsity(rating_df3)

#%% Impute numeric variables
# impute weight
weights = rating_df3.weight[rating_df3.weight.notnull()].str.replace('lbs', '').astype(int)
weight_mean = round(weights.sum() / weights.shape[0], 0)
rating_df3.weight[rating_df3.weight.isnull()] = 'nulllbs'
rating_df3.weight = rating_df3.weight.str.replace('lbs', '').str.replace('null', str(weight_mean))
rating_df3.weight = rating_df3.weight.astype(float)

# convert and impute height
heights = rating_df3.height[rating_df3.height.notnull()].str.replace('''"''', '').str.split("' ", expand=True)
heights2 = (heights[0].astype(int) * 12 + heights[1].astype(int))
height_mean = heights2.sum() / heights2.shape[0]
rating_df3.height[rating_df3.height.notnull()] = heights2
rating_df3.height[rating_df3.height.isnull()] = height_mean

# impute age
ages = rating_df3.age[rating_df3.age.notnull()].astype(int)
age_mean = ages.sum() / ages.shape[0]  # 33.8710
rating_df3.age[rating_df3.age.isnull()] = age_mean

#%% Get dummy
raw_df_cat = rating_df3[['user_id', 'item_id', 'fit', 'bust_size', 'rented_for', 'body_type', 'category']]
raw_df_dummy = pd.get_dummies(raw_df_cat, prefix=['user', 'item', 'fit', 'bust', 'rent', 'body', 'category'], dummy_na=True)

# raw_df_cat = rating_df3[['user_id', 'item_id']]
# raw_df_dummy = pd.get_dummies(raw_df_cat, prefix=['user', 'item'], dummy_na=True)

#%% Combine cleaned data and convert into matrix
# cleaned = pd.concat([rating_df3[['weight', 'height', 'age', 'size1', 'rating', 'review_date']], raw_df_dummy], axis=1, copy=False)
cleaned = pd.concat([rating_df3[['rating', 'review_date']], raw_df_dummy], axis=1, copy=False)

#%% Release memory
del raw_df_cat, raw_df_dummy

#%% Train-test split by date
cleaned.review_date = pd.to_datetime(cleaned.review_date,
                                     dayfirst=False,
                                     yearfirst=False
                                     )

cleaned.sort_values('review_date', axis=0, ascending=True, inplace=True)
cleaned.reset_index(inplace=True, drop=True)
cleaned.drop(['review_date'], axis=1, inplace=True)

boundary = round(len(cleaned) * 0.7)
train = cleaned.loc[:boundary]
test = cleaned.loc[boundary:]

# train.user_100157.value_counts()
# test.user_100157.value_counts()

#%% Kfold
train = shuffle(train)
boundary = [round(train.shape[0]*(0.1*ratio)) for ratio in range(2, 10, 2)]


kf = KFold(n_splits=5,shuffle=False)
for idx, (train_index , test_index) in enumerate(kf.split(train)):
    print(train.loc[train_index].shape, train.loc[test_index].shape)
    train.loc[train_index].to_csv('./data/train_fold{}.csv'.format(idx), header=False, index=False)
    train.loc[test_index].to_csv('./data/test_fold{}.csv'.format(idx), header=False, index=False)


#%% Save as csv
train.to_csv('./data/train_rating.csv', header=False, index=False)
test.to_csv('./data/test_rating.csv', header=False, index=False)

#%%



