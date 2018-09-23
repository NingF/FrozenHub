# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 22:59:35 2018

@author: Fwh_FrozenFire
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

df_concrete = pd.read_csv('concrete.csv')
print(df_concrete.head())
print(df_concrete.info())
df_concrete.columns = ['cement','slag','ash','water','superplastic','coarseagg','fineagg','age','strength']
y=df_concrete['strength']
X=df_concrete.drop(['strength'],axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
print("=================================================================")
###################EDA####################

#Pairplot
sns.set(style='whitegrid', context='notebook')
cols = ['cement','slag','ash','water','superplastic','coarseagg','fineagg','age','strength']
sns.pairplot(df_concrete[cols], size=1.9)
plt.show()
print("=================================================================")
#Heatmap
cm = np.corrcoef(df_concrete[cols].values.T)
sns.set(font_scale=1.5)
fig, hm = plt.subplots(figsize=(10,10))   
hm = sns.heatmap(cm, cbar=True,annot=True, square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
plt.figure(figsize=(40,40))
plt.show()
print("=================================================================")
#################Linear Regression##################
from sklearn.linear_model import LinearRegression
reg = LinearRegression(normalize = True)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
plt.plot(X_test, y_pred, color='black', linewidth=3)
plt.show()


#Residual plot
plt.scatter(y_pred,y_pred-y_test,	c='blue',marker='s',edgecolor='white',	label='Test data')
plt.xlabel('Predicted values LINEAR')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=0,xmax=100,color='black',lw=2)
plt.xlim([0,100])
plt.show()
print('Slope: %.3f' % reg.coef_[0],reg.coef_[1],reg.coef_[2],reg.coef_[3],reg.coef_[4]
        ,reg.coef_[5],reg.coef_[6],reg.coef_[7])
print('Y-Intercept: %.3f' % reg.intercept_)

print("R^2: {}".format(reg.score(X_test, y_test)))
mse = mean_squared_error(y_test,y_pred)
print("MSE: {}".format(mse))
print("=================================================================")
################Ridge###################
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(X_train,y_train)
y_predR = ridge.predict(X_test)

plt.scatter(y_predR,y_predR-y_test,	c='blue',marker='s',edgecolor='white',	label='Test data')
plt.xlabel('Predicted values RIDGE')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=0,xmax=100,color='black',lw=2)
plt.xlim([0,100])
plt.show()
print('Slope: %.3f' % ridge.coef_[0],ridge.coef_[1],ridge.coef_[2],ridge.coef_[3],ridge.coef_[4]
        ,ridge.coef_[5],ridge.coef_[6],ridge.coef_[7])
print('Y-Intercept: %.3f' % ridge.intercept_)

print("R^2: {}".format(ridge.score(X_test, y_test)))
mse = mean_squared_error(y_test,y_predR)
print("MSE: {}".format(mse))

print("=================================================================")
################Lasso###################
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
y_predL = lasso.predict(X_test)

plt.scatter(y_predL,y_predL-y_test,	c='blue',marker='s',edgecolor='white',	label='Test data')
plt.xlabel('Predicted values LASSO')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=0,xmax=100,color='black',lw=2)
plt.xlim([0,100])
plt.show()
print('Slope: %.3f' % lasso.coef_[0],lasso.coef_[1],lasso.coef_[2],lasso.coef_[3],lasso.coef_[4]
        ,lasso.coef_[5],lasso.coef_[6],lasso.coef_[7])
print('Y-Intercept: %.3f' % lasso.intercept_)

print("R^2: {}".format(lasso.score(X_test, y_test)))
mse = mean_squared_error(y_test,y_predL)
print("MSE: {}".format(mse))
print("=================================================================")
################Elastic###################
from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=0.1, l1_ratio=0.5)
elanet.fit(X_train,y_train)
y_predE = elanet.predict(X_test)

plt.scatter(y_predE,y_predE-y_test,	c='blue',marker='s',edgecolor='white',	label='Test data')
plt.xlabel('Predicted values ELASTICNET')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=0,xmax=100,color='black',lw=2)
plt.xlim([0,100])
plt.show()
print('Slope: %.3f' % elanet.coef_[0],elanet.coef_[1],elanet.coef_[2],elanet.coef_[3],elanet.coef_[4]
        ,elanet.coef_[5],elanet.coef_[6],elanet.coef_[7])
print('Y-Intercept: %.3f' % elanet.intercept_)

print("R^2: {}".format(elanet.score(X_test, y_test)))
mse = mean_squared_error(y_test,y_predE)
print("MSE: {}".format(mse))

###################################################
print("=================================================================")
print("My name is Ning Fan")
print("My NetID is: 673869376")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")









