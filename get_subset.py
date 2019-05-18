# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submission = pd.read_csv("../input/sample_submission.csv")

print("Training data size",train_data.shape)
print("test data size",test_data.shape)
submission.head()

train_data.head()

train_data['landmark_id'].value_counts().hist()

# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train_data.landmark_id.value_counts().head(100))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
print(temp)

temp_list = temp['landmark_id']
print(temp_list)

def in_list(x):
  return (x['landmark_id'] in temp_list.values)

unique = (train_data.apply(in_list, axis=1))

print((unique.unique()))

unique.head()

train_data['popular'] = unique
train_data.head()

train_data2 = train_data.drop(train_data[train_data.popular == False].index)

train_data2.head()

train_data2.shape

micro_brew = train_data2.head(400000)

micro_brew = micro_brew.drop(labels = 'popular', axis=1)

micro_brew.head()

micro_brew.to_csv('microbrew400.csv')