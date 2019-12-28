############################
# CHAPTER 6 - COMPRESSING DATA
# VIA DIMENSIONALITY REDUCTION
############################

############################
# IMPORTS
############################

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

############################
# STREAMLINE WORKFLOWS WITH PIPELINES
############################

# Loading Breast Cancer Wisconsin Data

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y) # using LabelEncoder to change M and B values numbers
print(y)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Combining Transformers and Estimators in a pipeline

# Creating a pipeline for StandardScalar, PCA and LogisticRegression

pipe_lr = Pipeline([
    ('scl', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', LogisticRegression(random_state=1))
])

pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


############################
# USING K-FOLDS CROSS-VALIDATION TO ASSESS MODEL PERFORMANCE
############################

kfold = StratifiedKFold(n_splits=10, random_state=1)
scores=[]
for k, (train, test) in enumerate(kfold.split(X_train, y_train)):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Using Scikit k-fold cross-validation scorer

scores = cross_val_score(estimator = pipe_lr, X = X_train, y  = y_train, cv = 10, n_jobs=1)
print('CV Accuracy Scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

############################
# DEBUGGING ALGORITHMS WITH LEARNING AND VALIDATION TECHNIQUES
############################



############################
# FINE-TUNING MACHINE LEARNING MODELS VIA GRID SEARCH
############################

############################
# LOOKING AT DIFFERENT PERFORMANCE EVALUATIONS METRICS
############################
