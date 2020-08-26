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


sns.countplot(x='Survived', data=df_train)
sns.countplot(x='Sex', data=df_train)
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train)


# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# In[ ]:


# Use pandas to figure out the proportion of women that survived, along with the proportion of men
print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Pclass'
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train)


# In[ ]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Embarked'
sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train)


# In[ ]:


# Use seaborn to plot a histogram of the 'Fare' column of df_train
sns.distplot(df_train.Fare, kde=False)


# In[ ]:


# Use a pandas plotting method to plot the column 'Fare' for each value of 'Survived' on the same plot.
df_train.groupby('Survived').Fare.hist(alpha=0.6)


# In[ ]:


# Use seaborn to plot a histogram of the 'Age' column of df_train. You'll need to drop null values before doing so
df_train_drop = df_train.dropna()
sns.distplot(df_train_drop.Age, kde=False)


# In[ ]:


sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.3, jitter=True)
sns.swarmplot(x='Survived', y='Fare', data=df_train)


# In[ ]:


#Use the DataFrame method .describe() to check out summary statistics of 'Fare' as a function of survival
df_train.groupby('Survived').Fare.describe()


# In[ ]:


# Use seaborn to plot a scatter plot of 'Age' against 'Fare', colored by 'Survived'
sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.5})


# In[ ]:


# Use seaborn to create a pairplot of df_train, colored by 'Survived'
sns.pairplot(df_train_drop, hue='Survived')


# In[ ]:


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# Check out your new DataFrame data using the info() method
data.info()


# In[ ]:


# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()


# In[ ]:


# Encode the data with numbers because most machine learning models might require numerical inputs
# yo can do this using Pandas function get_dummies() which converts the categorical variable into numerical
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# In[ ]:


# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[ ]:


data.info()


# In[ ]:


data_train = data.iloc[:891]
data_test = data.iloc[891:]
# A Scikit requirement transform the dataframes to arrays
X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


# build your decision tree classifier with max_depth=3 and then fit it your data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[ ]:


# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('1st_dec_tree.csv', index=False)


# In[ ]:


# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# Extract Title from Name, store in column and plot barplot
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data)
plt.xticks(rotation=45)


# In[ ]:


data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data)
plt.xticks(rotation=45)


# In[ ]:


# Did they have a Cabin?
data['Has_Cabin'] = ~data.Cabin.isnull()

# Drop columns and view head
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()


# In[ ]:


# Impute missing values for Age, Fare, Embarked
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())
data['Embarked'] = data['Embarked'].fillna('S')
data.info()


# In[ ]:


# Binning numerical columns
data['CatAge'] = pd.qcut(data.Age, q=4, labels=False )
data['CatFare']= pd.qcut(data.Fare, q=4, labels=False)
data.head()


# In[ ]:


data = data.drop(['Age', 'Fare','SibSp','Parch'], axis=1)
data.head()


# In[ ]:


# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()


# In[ ]:


# Split into test.train
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values

# Setup the hyperparameter grid
dep = np.arange(1,9)
param_grid = {'max_depth' : dep}

# Instantiate a decision tree classifier: clf
clf = tree.DecisionTreeClassifier()

# Instantiate the GridSearchCV object: clf_cv
clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)

# Fit it to the data
clf_cv.fit(X, y)

# Print the tuned parameter and score
print("Tuned Decision Tree Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))


# In[ ]:


# Now, you can make predictions on your test set, create a new column 'Survived' and store your predictions in it
Y_pred = clf_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('dec_tree_feat_eng.csv', index=False)


# In[ ]:


# import logisitc regression from sklearn
from sklearn.linear_model import LogisticRegression

#instantiate the classifier without any parameters
logreg = LogisticRegression()

#fit the data to the classifier
logreg.fit(X,y)

#predict the survivors and submit the results
Y_pred = logreg.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('log_reg_feat_eng.csv', index=False)


# In[ ]:


c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv=5)

# Fit it to the training data
logreg_cv.fit(X,y)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

#predict the survivors and submit the results
Y_pred = logreg_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('log_reg__feat_eng.csv', index=False)


# In[ ]:


#import random forest classifer
from sklearn.ensemble import RandomForestClassifier

#instantiate RandomForest
rf_clf = RandomForestClassifier()

#fit the data
rf_clf.fit(X,y)

#predict the survivors and submit the results
Y_pred = rf_clf.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('random_forest_feat_eng.csv', index=False)


# In[ ]:


# create parameter grid for hyperparameter tuning
n_estimators = np.arange(10,50)
params_grid = {'n_estimators':n_estimators}

#instantiate RandomForest
rf_clf = RandomForestClassifier()

# Instantiate the GridSearchCV object
rf_clf_cv = GridSearchCV(rf_clf,params_grid,cv=5)

# Fit it to the training data
rf_clf_cv.fit(X,y)

# Print the optimal parameters and best score
print("Tuned Random Forest Classifier Parameter: {}".format(rf_clf_cv.best_params_))
print("Tuned Random Forest Classifier Accuracy: {}".format(rf_clf_cv.best_score_))

#predict the survivors and submit the results
Y_pred = rf_clf_cv.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('random_forest_cv_feat_eng.csv', index=False)


# In[ ]:




