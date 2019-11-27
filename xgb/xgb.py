#!/usr/bin/env python
# coding: utf-8

# ## Naive Modeling using Minimum Analysis
# 
# **Reference** : [Link](https://www.kaggle.com/kimchiwoong/simple-fast-check-xgboost-prediction-performance)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')


# ## 1. Loading the Dataset

# ### 1-1. Reducing Memory Usage via Downsizing Datatypes
# Reducing memory usage by downsizing datatypes in the dataset is now a common practice since it speeds up the overall process :)

# In[2]:


train_tr = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
test_tr = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')


# In[3]:


def reduce_mem(df):
  start_mem=df.memory_usage().sum()/1024**2
  print('Initial Memory Usage : {:.2f} MB'.format(start_mem))
  for col in df.columns:
    col_type=df[col].dtype
    if col_type != object:
      mn, mx = df[col].min(), df[col].max()
      if str(col_type)[:3]=='int':
        if mn>np.iinfo(np.int8).min and mx<np.iinfo(np.int8).max:
          df[col]=df[col].astype(np.int8)
        elif mn>np.iinfo(np.int16).min and mx<np.iinfo(np.int16).max:
          df[col]=df[col].astype(np.int16)
        elif mn>np.iinfo(np.int32).min and mx<np.iinfo(np.int32).max:
          df[col]=df[col].astype(np.int32)
      else:
        if mn>np.finfo(np.float16).min and mx<np.finfo(np.float16).max:
          df[col]=df[col].astype(np.float16)
        elif mn>np.finfo(np.float32).min and mx<np.finfo(np.float32).max:
          df[col]=df[col].astype(np.float32)
  end_mem = df.memory_usage().sum()/1024**2
  print('Final Memory Usage : {:.2f} MB'.format(end_mem))
  print('Decreased by {:.2f}%'.format(100*(start_mem-end_mem)/start_mem))
  return df


# In[4]:


print(train_tr.info())
print(train_id.info())


# In[5]:


train_tr = reduce_mem(train_tr)
train_id = reduce_mem(train_id)


# In[6]:


print(train_tr.info())
print(train_id.info())


# In[7]:


import gc
gc.collect()
test_tr = reduce_mem(test_tr)
test_id = reduce_mem(test_id)


# In[8]:


print(test_tr.info())
print(test_id.info())


# ### 1-2. Joining the Datasets
# 
# Joining the divided datasets eases the afterall process from EDA to predictive modeling.

# In[9]:


train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
dataset = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)


# In[10]:


train_len = len(train)
del train_tr, train_id, test_tr, test_id
gc.collect()


# ## 2. Exploratory Data Analysis

# ### 2-1. Missing Data

# In[11]:


def missing_data(df):
    count = df.isnull().sum()
    percent = df.isnull().sum()/df.isnull().count()*100
    total = pd.concat([count, percent], axis=1, keys=['Count', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    total['dtypes'] = types
    return np.transpose(total)


# In[12]:


missing_df = missing_data(dataset)
missing_df


# ### 2-2. Numerical Columns

# In[13]:


num_cols = [col for col in dataset.columns if dataset[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]
dataset[num_cols].describe()


# ### 2-3. Categorical Columns

# In[14]:


cat_cols = [col for col in dataset.columns if dataset[col].dtype not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']]
dataset[cat_cols].describe()


# In[15]:


for col in cat_cols:
    print('-'*25+'['+col+']'+'-'*25)
    print(dataset[[col, 'isFraud']].groupby(col).mean()*100)


# ## 3. Feature Engineering

# In[16]:


print(dataset.shape)


# ### 3-1. Adding Datetimes

# In[17]:


import datetime

genesis = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d')
dataset['Date'] = dataset['TransactionDT'].apply(lambda x : genesis+datetime.timedelta(seconds=x))


# In[18]:


dataset['Weekdays'] = dataset['Date'].dt.dayofweek
dataset['Days'] = dataset['Date'].dt.day
dataset['Hours'] = dataset['Date'].dt.hour


# In[19]:


fig, ax = plt.subplots(1, 3, figsize=(15, 5))

g = sns.barplot(dataset[dataset.index<train_len].Weekdays, train.isFraud, ax=ax[0])
ax[0].set_title('Fraud Charges by Weekdays')
plt.setp(g.get_xticklabels(), visible=False)

g = sns.barplot(dataset[dataset.index<train_len].Days, train.isFraud, ax=ax[1])
ax[1].set_title('Fraud Charges by Days')
plt.setp(g.get_xticklabels(), visible=False)

g = sns.barplot(dataset[dataset.index<train_len].Hours, train.isFraud, ax=ax[2])
ax[2].set_title('Fraud Charges by Hours')
plt.setp(g.get_xticklabels(), visible=False)

plt.show()


# In[20]:


dataset.drop('Date', axis=1, inplace=True)


# ### 3-2. Handling Rare or Missing Email Domains

# In[21]:


print(dataset['P_emaildomain'].value_counts().head())
print('Data type : {}'.format(dataset['P_emaildomain'].dtype))


# In[22]:


dataset.loc[(dataset.P_emaildomain!='gmail.com')&(dataset.P_emaildomain!='yahoo.com')&(dataset.P_emaildomain!='hotmail.com')&(dataset.P_emaildomain!='anonymous.com')&(dataset.P_emaildomain!='aol.com'), 'P_emaildomain'] = 'etc'


# In[23]:


sns.countplot(dataset['P_emaildomain'])
fig = plt.gcf()
fig.set_size_inches(10, 4)
plt.show()


# In[24]:


print(dataset['R_emaildomain'].value_counts().head())
print('Data type : {}'.format(dataset['P_emaildomain'].dtype))


# In[25]:


dataset.loc[(dataset.R_emaildomain!='gmail.com')&(dataset.R_emaildomain!='hotmail.com')&(dataset.R_emaildomain!='anonymous.com')&(dataset.R_emaildomain!='yahoo.com')&(dataset.R_emaildomain!='aol.com'), 'R_emaildomain'] = 'etc'


# In[26]:


sns.countplot(dataset['R_emaildomain'])
fig = plt.gcf()
fig.set_size_inches(10, 4)
plt.show()


# ### 3-3. Operating Systems

# In[27]:


top_os = dataset[['id_30', 'isFraud']].groupby(['id_30']).mean().sort_values(by=['isFraud'], ascending=False).head(10)
top_os.T


# It seems obvious that fraud charges were mostly run by mobile devices or devices that run rare operating systems(denoted by *other*)

# In[28]:


top_os = list(top_os.index)


# In[29]:


all_os = list(dataset['id_30'].unique())
safe_os = [os for os in all_os if os not in top_os]


# In[30]:


dataset.id_30.replace(safe_os, 'etc', inplace=True)


# In[31]:


dataset[['id_30', 'isFraud']].groupby(['id_30']).mean().T


# ### 3-4. Browsers

# In[32]:


top_browsers = dataset[['id_31', 'isFraud']].groupby(['id_31']).mean().sort_values(by=['isFraud'], ascending=False).head(10)
top_browsers.T


# In[33]:


top_browsers = list(top_browsers.index)


# In[34]:


all_browsers = list(dataset['id_31'].unique())
safe_browsers = [brw for brw in all_browsers if brw not in top_browsers]


# In[35]:


dataset.id_31.replace(safe_browsers, 'etc', inplace=True)


# In[36]:


dataset[['id_31', 'isFraud']].groupby('id_31').mean().sort_values(by='isFraud', ascending=False).T


# ### 3-5. Screen Sizes
# 
# Screen sizes can be factors to track specific device types

# In[37]:


top_scrsz = dataset[['id_33', 'isFraud']].groupby(['id_33']).mean().sort_values(by=['isFraud'], ascending=False).head(15)
top_scrsz.T


# In[38]:


top_scrsz = list(top_scrsz.index)


# In[39]:


all_scrsz = dataset['id_33'].unique()
safe_scrsz = [s for s in all_scrsz if s not in top_scrsz]


# In[40]:


dataset.id_33.replace(safe_scrsz, 'etc', inplace=True)


# In[41]:


dataset[['id_33', 'isFraud']].groupby('id_33').mean().sort_values(by='isFraud', ascending=False).T


# ### 3-6. Device Information

# In[42]:


top_dev = dataset[['DeviceInfo', 'isFraud']].groupby(['DeviceInfo']).mean().sort_values(by='isFraud', ascending=False).head(10)
top_dev.T


# In[43]:


top_dev = list(top_dev.loc[top_dev.isFraud>0.5].index)
top_dev


# In[44]:


all_dev = dataset['DeviceInfo'].unique()
safe_dev = [dev for dev in all_dev if dev not in top_dev]


# In[45]:


dataset.DeviceInfo.replace(safe_dev, 'etc', inplace=True)


# In[46]:


dataset[['DeviceInfo', 'isFraud']].groupby('DeviceInfo').mean().sort_values(by=['isFraud'], ascending=False).T


# ### 3-7. One Hot Encoding

# In[47]:


dataset_num = dataset.select_dtypes(exclude=['object'])
dataset_num.head()


# In[48]:


dataset_cat = dataset.select_dtypes(include=['object'])
dataset_cat.head()


# In[49]:


print('Added Columns : {}'.format(sum(dataset_cat.nunique().values)-len(dataset_cat.columns)))


# In[50]:


dataset_cat_new = pd.get_dummies(dataset_cat)
dataset = pd.concat([dataset_num, dataset_cat_new], axis=1)
dataset.shape


# In[51]:


dataset.drop('TransactionID', axis=1, inplace=True)
del dataset_num, dataset_cat
gc.collect()


# ## 4. Predictive Modeling

# In[52]:


dataset.head()


# In[53]:


train = dataset[:train_len]
test = dataset[train_len:]


# In[54]:


y = train.isFraud
X = train.drop('isFraud', axis=1)
test_X = test.drop('isFraud', axis=1)


# In[55]:


np.unique(y)


# In[56]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)


# ### 4-1. Extreme Gradient Boosting(XGB)

# In[57]:


import xgboost as xgb

hyper = {
    'booster' : 'gbtree',
    'max_depth' : 6,
    'nthread' : -1,
    'num_class' : 1,
    'objective' : 'binary:logistic',
    'verbosity' : 0,
    'eval_metric' : 'auc',
    'eta' : 0.1,
    'tree_method' : 'gpu_hist',
    'min_child_weight' : 1,
    'colsample_bytree' : 0.8,
    'colsample_bylevel' : 0.8,
    'seed' : 0
}

dtrn = xgb.DMatrix(train_X, label=train_y, feature_names=X.columns)
dval = xgb.DMatrix(val_X, label=val_y, feature_names=X.columns)

_xgb = xgb.train(hyper, dtrn, num_boost_round=10000, evals=[(dtrn, 'train'), (dval, 'eval')], early_stopping_rounds=200, verbose_eval=200)


# In[58]:


dtst = xgb.DMatrix(test_X, feature_names=X.columns)
preds_xgb = _xgb.predict(dtst)


# ### 4-2. Light Gradient Boosting Machine(LGBM)

# In[59]:


import lightgbm as lgbm

hyper = {
    'num_leaves' : 500,
    'min_child_weight': 0.03,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.4,
    'min_data_in_leaf': 100,
    'objective': 'binary',
    'max_depth': 6,
    'learning_rate': 0.05,
    'boosting_type': 'gbdt',
    'bagging_seed': 10,
    'metric': 'auc',
    'verbosity': 0,
    'reg_alpha': 0.4,
    'reg_lambda': 0.6,
    'random_state': 0
}

dtrain = lgbm.Dataset(train_X, label=train_y)
dvalid = lgbm.Dataset(val_X, label=val_y)
model = lgbm.train(hyper, dtrain, 10000, valid_sets=[dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)


# In[60]:


preds_lgb = model.predict(test_X)


# ## 5. Submission

# In[61]:


submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
submission['isFraud'] = np.nan
submission.head()


# In[62]:


submission['isFraud'] = (0.5*preds_xgb)+(0.5*preds_lgb)
submission.head()


# In[63]:


submission.to_csv('submission_mark_1.csv', index=False)

