
print("My name is Tianhao Zhao")
print("My NetID is: tzhao25")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print(" ")
## Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import numpy as np


## Read Housing Data 
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',
                 header=None,sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

## Exploratory Data Analysis
print(df.head())
print(df.describe())
corMat = DataFrame(df.corr())
plt.pcolor(corMat)
plt.show()

##Split data into training and test sets 
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

##Linear model 
from sklearn.linear_model import LinearRegression
# Import LinearRegression
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
print('Slope: %.3f' % slr.coef_[0])

print('Intercept: %.3f' % slr.intercept_)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('Linear Model')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)

plt.show()
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

##Ridge Model 
from sklearn.linear_model import Ridge
ridge1 = Ridge(alpha=1)
ridge1.fit(X_train, y_train)
y_train_pred = ridge1.predict(X_train)
y_test_pred = ridge1.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('Ridge Model with alpha = 1')
plt.xlabel('Predicted values')
plt.ylabel('Residuals at alpha =1.0')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.show()
print('At alpha =1.0,MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha =1.0,R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

ridge2 = Ridge(alpha=0.5)
ridge2.fit(X_train, y_train)
y_train_pred = ridge2.predict(X_train)
y_test_pred = ridge2.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('Ridge Model with alpha = 0.5')
plt.xlabel('Predicted values')
plt.ylabel('Residuals at alpha =1.0')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.show()
print('At alpha =0.5,MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha =0.5,R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

ridge3 = Ridge(alpha=0.1)
ridge3.fit(X_train, y_train)
y_train_pred = ridge3.predict(X_train)
y_test_pred = ridge3.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('Ridge Model with alpha = 0.1')
plt.xlabel('Predicted values')
plt.ylabel('Residuals at alpha =1.0')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-20, xmax=50, color='black', lw=2)
plt.show()
print('At alpha =0.1,MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha =0.1,R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))
##At alpha =0.1 gives the best performing model 

## LASSO Model
from sklearn.linear_model import Lasso
lasso1 = Lasso(alpha=1.0)
lasso1.fit(X_train, y_train)
y_train_pred = lasso1.predict(X_train)
y_test_pred = lasso1.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.title('LASSO Model with alpha = 1')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-20, xmax=50, color='black', lw=2)
plt.show()
print('At alpha = 1.0, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha = 1.0, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

lasso2 = Lasso(alpha=0.5)
lasso2.fit(X_train, y_train)
y_train_pred = lasso2.predict(X_train)
y_test_pred = lasso2.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('LASSO Model with alpha = 0.5')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.show()
print('At alpha = 0.5, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha = 0.5, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

lasso3 = Lasso(alpha=0.1)
lasso3.fit(X_train, y_train)
y_train_pred = lasso3.predict(X_train)
y_test_pred = lasso3.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('LASSO Model with alpha = 0.1')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.show()
print('At alpha = 0.1, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha = 0.1, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

## alpha = 0.1 gives the best performing model 

## Elastic Net Model
from sklearn.linear_model import ElasticNet
elanet1 = ElasticNet(alpha=1.0, l1_ratio=0.5)
elanet1.fit(X_train, y_train)
y_train_pred = elanet1.predict(X_train)
y_test_pred = elanet1.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('Elastic Net Model with alpha = 1')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.show()
print('At alpha = 1.0, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha = 1.0, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


elanet2 = ElasticNet(alpha=0.1, l1_ratio=0.5)
elanet2.fit(X_train, y_train)
y_train_pred = elanet2.predict(X_train)
y_test_pred = elanet2.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('Elastic Net Model with alpha = 0.1')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.show()
print('At alpha = 0.1, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha = 0.1, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))


elanet3 = ElasticNet(alpha=0.05, l1_ratio=0.5)
elanet3.fit(X_train, y_train)
y_train_pred = elanet3.predict(X_train)
y_test_pred = elanet3.predict(X_test)
plt.scatter(y_train_pred, y_train_pred - y_train, c='green', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='red', marker='s', edgecolor='white', label='Test data')
plt.title('Elastic Net Model with alpha = 0.05')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.show()
print('At alpha = 0.05, MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('At alpha = 0.05, R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred),r2_score(y_test, y_test_pred)))

## alpha=0.05 gives the best performing model 

