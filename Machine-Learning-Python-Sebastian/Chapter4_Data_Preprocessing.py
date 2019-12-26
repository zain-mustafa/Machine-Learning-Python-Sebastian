############################
# CHAPTER 4 - BUILDING GOOD TRAINING SETS
# DATA PREPROCESSING
############################

#IMPORTS

import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

############################
# DEALING WITH MISSING VALUES
############################

csv_data = '''
A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
'''

df = pd.read_csv("exampleCSV.csv")

# The dataframe will replace the empty cells with NaN
print(df)

# Using the isnull ,ethod to return a DataFrame that contains a missing cell
print(df.isnull().sum())

print(df.values)

# Dropping feature that have empty values (Rows)
print(df.dropna())

# Dropping Columns that have atleast one null value
print(df.dropna(axis=1))

# Only drop columns which are completely null
print(df.dropna(how='all'))

# drop rows that at least 4 no-NaN values
df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

print(df)

############################
# IMPUTING MISSING VALUES
############################

imr = SimpleImputer(missing_values=np.nan, strategy="mean")

imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)

############################
# HANDLING CATEGORICAL DATA
############################

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1'],
])

df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

# Mapping Ordinal Features

size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
    }

df['size'] = df['size'].map(size_mapping)
print(df)

# Mapping the Values back to originals

inv_size_mapping = {
    v: k for k, v in size_mapping.items()
}

df['size'] = df['size'].map(inv_size_mapping)

print(df)

# Encoding Class Labels

class_mapping = {
    label: idx for idx,
    label in enumerate(np.unique(df['classlabel']))
}

print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)

print(df)

# Reversing Class enconding

inv_class_mapping = {
    v: k for k, v in class_mapping.items()
}

df['classlabel'] = df['classlabel'].map(inv_class_mapping)

print(df)

# Using Scikit learn LabelEncoder for mapping

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

z = class_le.inverse_transform(y)
print(z)

# Performing one-hot encoding on nominal features

X = df[['color', 'size', 'price']].values
print(X)

color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)

ohe = OneHotEncoder(categories='auto')
print(ohe.fit_transform(X).toarray())

# Using get dummies to implement one-hot encoding

print(pd.get_dummies(df[['price', 'color', 'size']]))

############################
# PARTITIONING A DATASET IN TRAINING AND TEST SETS
############################

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
print(df_wine)

df_wine.columns = ['Class label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash',
'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
'Color Intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

print('Class Labels', np.unique(df_wine['Class label']))

print(df_wine.head())

X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


############################
# BRINGING FEATURES ONTO THE SAME SCALE
############################

# Normalization Using Min-Max Scaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

print(X_train_norm)
print(X_test_norm)

# Standardization Using Standard Scalar

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print(X_train_std)
print(X_test_std)

############################
# SELECTING MEANINGFUL FEATURES
############################

# Sparse solutions with L1 Regularization

LogisticRegression(penalty='l1')

lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy: ', lr.score(X_test_std, y_test))

print(lr.intercept_) # Intercepts

print(lr.coef_) # Weight coefficients for each of the classes' 13 features

# Plotting the regularization path

fig = plt.figure()
ax = plt.subplot(111)

colors = [
    'blue',
    'green',
    'red',
    'cyan',
    'magenta',
    'yellow',
    'black',
    'pink',
    'lightgreen',
    'lightblue',
    'gray',
    'indigo',
    'orange'
    ]

weights, params = [], []

for c in np.arange(-4, 6, dtype=float):
    lr = LogisticRegression(penalty='l1',C=10**c,random_state=0)
    lr.fit(X_train_std, y_train)
    print('###########')
    print(lr.coef_)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)
print(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:,column], label=df_wine.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight cocefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend('upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()

# Sequential Backward Selection for feature selection:
# Applys a greedy algorithm to sequentially remove feature from the real feature space
# until the set number of features is reached.

############################
# ASSESSING FEATURE IMPORTANCE WITH RANDOM FOREST
############################

# Using feature_importance_ attribute from Scikit

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

