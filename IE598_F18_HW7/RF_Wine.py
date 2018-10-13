# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 21:06:47 2018

@author: Fwh_FrozenFire
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score

df_wine = pd.read_csv('wine.csv')
df_wine.columns = ['Alcohol','Malic acid','ash','Alcalinity of ash','Magnesium','Total phenols',
               'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity',
               'Hue','OD280/OD315 of diluted wines','Proline','Class']
y=df_wine['Class']
X1=df_wine.drop(['Class'],axis = 1)

#Normalize the data to 0-10
X= ((X1-X1.min())/(X1.max()-X1.min()))*10
#Hold out 10% data for out-of-sample test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=42)

################################Random Forest#############################
from sklearn.ensemble import RandomForestClassifier

for i in [10,50,100,200,500,1000]:
    rf = RandomForestClassifier(n_estimators= i, random_state=2)
    rf.fit(X_train,y_train)
    scores = cross_val_score(estimator = rf, X=X_train,y=y_train,cv=10,n_jobs=1)
    print(i)
    print('CV accuracy for N_estimators = ',i, ': %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('Out-of-sample Accuracy Score for N_estimators = ',i, ':', rf.score(X_test,y_test))
print('======================================================================')

##########################Feature Importance###############################
rf = RandomForestClassifier(n_estimators= 50, random_state=2)
rf.fit(X_train,y_train)
importances = pd.Series(data=rf.feature_importances_, index= X_train.columns)
importances_sorted = importances.sort_values()
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()

print('=================================================')


print("My name is Ning Fan")
print("My NetID is: 673869376")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")