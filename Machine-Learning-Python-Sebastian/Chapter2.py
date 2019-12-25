
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Chapter3 import plot_decision_regions

class Perceptron(object):
    """
    Perceptron Classifier
    
    Parameters
    ------------
    eta: float
        Learning Rate (between 0 and 1.0)
        
    n_iter: int
        Passes over the training set (epoch)
        
    Attributes
    ------------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Number of misclassifications in every epoch.
    
    """
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        
        """
        Fit Training Data
        
        Parameters
        ------------
        X: {array-like}, shape = [n_samples, n_features]
            Training Vectors, where n_samples is the number of samples and n_features is the number of features.
        
        y: {array-like}, shape = [n_samples]
            Target Values
            
        Returns
        ---------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1]) # Generating a zeros Matrix with the column length of the X Matrix
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate Net Input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


#Loading the CSV
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#Printing the last 5 Entries
df.head()


y = df.iloc[0:100, 4].values # Getting the values from the dataframe
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1 ), ppn.errors_, marker="o")
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.show()


# 
# ## Adaptive Linear Neurons


class AdalineGD(object):
    """ADAptive LInear NEuron classifier
    
     Parameters
    ------------
    eta: float
        Learning Rate (between 0 and 1.0)
        
    n_iter: int
        Passes over the training set (epoch)
        
    Attributes
    ------------
    w_: 1d-array
        Weights after fitting.
    errors_: list
        Number of misclassifications in every epoch.
    """
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        
        """
        Fit Training Data
        
        Parameters
        ------------
        X: {array-like}, shape = [n_samples, n_features]
            Training Vectors, where n_samples is the number of samples and n_features is the number of features.
        
        y: {array-like}, shape = [n_samples]
            Target Values
            
        Returns
        ---------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1]) # Generating a zeros Matrix with the column length of the X Matrix
        self.cost_ = []
        
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate Net Input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()


# ## Standardization


X_std = np.copy(X)
X_std[:, 0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:, 1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


# ## Applying Adaline after Standardization


ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()



