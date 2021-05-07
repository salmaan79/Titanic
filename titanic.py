# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:32:47 2020

@author: Salmaan Ahmed Ansari
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:58:28 2020

@author: Salmaan Ahmed Ansari
"""


# Importing the libraries
import numpy as np

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the dataset
dataset = pd.read_csv('train.csv', sep = ',')

X = dataset.iloc[:, [1,3,4,5,6,8,10]].values

y = dataset.iloc[:, 11:12].values
print(X)
print(y)
dataset.info()



# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, [0,1,3,4,6]])
X[:, [0,1,3,4,6]] = imputer.transform(X[:, [0,1,3,4,6]])
print(X)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [2,5]])
X[:, [2,5]] = imputer.transform(X[:, [2,5]])
print(X)


"""#taking care of missing data
dataset['Property_Area'].fillna(dataset['Property_Area'].mode()[0], inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].median(), inplace=True)
"""

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X[:, 1] = le_X.fit_transform(X[:, 1])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]


y=y.astype('int')


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

seed= 42
classifier =RandomForestClassifier(n_estimators=100000, criterion='entropy', max_depth=5, min_samples_split=2,
                           min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, 
                           n_jobs=1, random_state=seed,verbose=0)

classifier.fit(X, y)


result_train = classifier.score(X, y)

"""
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000], 'kernel': ['rbf']},
              {'C': [1, 1.5, 2, 3, 4, 5, 6], 'kernel': ['rbf'], 'gamma': [0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
"""





#for test dataset

dataset_test = pd.read_csv('test.csv')
X_tes = dataset_test.iloc[:, [1,3,4,5,6,8,10]].values


# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X_tes[:, [0,1,3,4,6]])
X_tes[:, [0,1,3,4,6]] = imputer.transform(X_tes[:, [0,1,3,4,6]])
print(X_tes)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_tes[:, [2,5]])
X_tes[:, [2,5]] = imputer.transform(X_tes[:, [2,5]])
print(X_tes)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
X_tes[:, 1] = le_X.fit_transform(X_tes[:, 1])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
X_tes = np.array(ct.fit_transform(X_tes))
X_tes = X_tes[:, 1:]

# Feature Scaling
X_tes = sc.transform(X_tes)




# Predicting the Test set results
y_tes = classifier.predict(X_tes)

y_tes



