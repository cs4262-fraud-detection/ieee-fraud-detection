#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing, decomposition, model_selection, linear_model, metrics, ensemble, svm, utils
from sklearn.datasets import make_classification
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import metrics
import sklearn


# In[2]:


# Get the data
transactions = pd.read_csv('../data/train_transaction.csv')
identities = pd.read_csv('../data/train_identity.csv')


# In[3]:


# Join datasets
dataset = transactions.merge(identities, how='left', left_index=True, right_index=True)


# In[4]:


# Reduce memory usage
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




# TODO: Add source for the reduce_mem code snippet? Should we include this on final submission?


# In[5]:


dataset = reduce_mem(dataset)


# In[6]:


del transactions, identities
gc.collect()


# In[7]:


# TODO: Display some sort of visualization of the data (e.g. show that is imbalanced)?


# In[8]:


'''
not_fraud_downsampled = utils.resample(not_fraud, replace=False, n_samples = len(fraud), random_state = 27) 
sownsampled_dataset = pd.concat([not_fraud_downsampled, fraud])
'''
# Note: Downsampling does not seem to improve model results, so I don't think it's worth doing it. However, we
# should definitely mention on the final presentation/paper that we tried it.


# In[9]:


# TODO: Drop columns with too many NaN values?
'''
def drop_columns(dataset, threshold):
    columns_to_drop = []
    for column in dataset.columns:
        if dataset[column].isna().sum()/dataset.shape[0] < threshold:
            columns_to_drop.append(column)
    return dataset.drop(columns_to_drop, axis=1)
'''


# In[10]:


# Fill NaN values
dataset = dataset.fillna(0)


# In[11]:


# Encode labels
for col in dataset.columns:
    if dataset[col].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(dataset[col].values))
        dataset[col] = lbl.transform(list(dataset[col].values))


# In[12]:


y = dataset.isFraud.values
dataset = dataset.drop('isFraud',axis=1)


# In[13]:


# Standardize the data
dataset_columns = list(dataset.columns)
dataset[dataset_columns] = preprocessing.StandardScaler().fit_transform(dataset[dataset_columns])


# In[14]:


X_pre_pca = dataset[dataset_columns]


# In[15]:


# Reduce number of dimensions through PCA
reduced_number_of_dimensions = 15 # Note: This number should be increased before making the final
# submission. Lower values lead to faster model training but lower model accuracy/f1-scores.
pca = decomposition.PCA(n_components=reduced_number_of_dimensions, random_state=42)
X_post_pca = pca.fit_transform(X_pre_pca) 
print(pca.explained_variance_ratio_.sum())


# In[16]:


# Perform 80/20 train/test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_post_pca, y, test_size=0.2, random_state=1)


# In[17]:


# The following function returns a model with the hyperparameters that yield the best f1 score
def get_model_with_best_estimators(model, parameters, X_train, y_train):
    grid_search = model_selection.GridSearchCV(model, parameters, scoring='f1', cv=4)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# In[18]:


random_forest_parameters = {'n_estimators':[50, 100, 200], 'criterion':('gini', 'entropy')}
random_forest = ensemble.RandomForestClassifier(n_jobs=-1)
best_random_forest_model = get_model_with_best_estimators(random_forest, random_forest_parameters, X_train, y_train)


# In[ ]:


logistic_regression_parameters = {'C':[1, 10], 'penalty':('l2',), 'solver':('saga', 'newton-cg')}
logistic_regression = linear_model.LogisticRegression(n_jobs=-1, max_iter=200)
best_logistic_regression_model = get_model_with_best_estimators(logistic_regression, logistic_regression_parameters, X_train, y_train)


# In[ ]:


svm_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svm = svm.SVC(max_iter=200)
best_svm_model = get_model_with_best_estimators(svm, svm_parameters, X_train, y_train)


# In[ ]:


print(sklearn.metrics.classification_report(y_test, best_random_forest_model.predict(X_test)))


# In[ ]:


print(sklearn.metrics.classification_report(y_test, best_logistic_regression_model.predict(X_test)))


# In[ ]:


print(sklearn.metrics.classification_report(y_test, best_svm_model.predict(X_test)))


# In[20]:


def split_with_PCA(k, x_tr, y):
    X_pca = decomposition.PCA(n_components=k).fit_transform(x_tr)  
    return model_selection.train_test_split(X_pca, y, test_size=.2, random_state=1)


# In[21]:


# Neural network approach

K = [50, 100, 150, 200, 250, 300, 350, 400]
nns = [Sequential() for _ in range(len(K))]
results = []
batch_size = 5000
num_epochs = 10

for k, cur_nn in zip(K, nns):
    n_cols = k
    x_tr, x_test, y_tr, y_test = split_with_PCA(k, X_pre_pca, y)
    cur_nn.add(Dense(300, activation='relu', input_shape=(n_cols,)))
    cur_nn.add(Dropout(0.2))
    cur_nn.add(Dense(500, activation='relu'))
    cur_nn.add(Dropout(0.2))
    cur_nn.add(Dense(100, activation='relu'))
    cur_nn.add(Dropout(0.2))
    cur_nn.add(Dense(25, activation='relu'))
    cur_nn.add(Dropout(0.2))
    cur_nn.add(Dense(1, activation='sigmoid'))
    cur_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])
    x_vl, y_vl = x_tr[:batch_size], y_tr[:batch_size]
    x_train, y_train = x_tr[batch_size:], y_tr[batch_size:]
    print("For k=" + str(k))
    cur_nn.fit(x_train, y_train, validation_data=(x_vl, y_vl), epochs=num_epochs, batch_size=batch_size)
    # res = cur_nn.evaluate(x_test, y_test, batch_size=128, verbose=0)
    print(sklearn.metrics.classification_report(y_test, cur_nn.predict_classes(x_test)))
    # results.append(res)
    # print('test loss, test acc, categorical accuracy:', res)


# In[ ]:




