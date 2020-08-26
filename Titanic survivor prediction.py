#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# View first few lines of training data
df_train.head()


# In[ ]:


# You can also view the data types and missing data
df_train.info()
# you can also see the statistical summary of the training data
df_train.describe()


# In[ ]:


sns.countplot(x='Survived', data=df_train);
sns.countplot(x='Sex', data=df_train);
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);


# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


# Use pandas to figure out the proportion of women that survived, along with the proportion of men
print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Pclass'
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);


# In[ ]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Embarked'
sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);


# In[ ]:


# Use seaborn to plot a histogram of the 'Fare' column of df_train
sns.distplot(df_train.Fare, kde=False);

