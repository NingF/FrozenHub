# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 22:22:34 2018

@author: Fwh_FrozenFire
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split

pd.set_option('display.max_columns',15)

df_wine = pd.read_csv('wine.csv')
print(df_wine.head())
print(df_wine.info())

df_wine.columns = ['Alcohol','Malic acid','ash','Alcalinity of ash','Magnesium','Total phenols',
               'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity',
               'Hue','OD280/OD315 of diluted wines','Proline','Class']
y=df_wine['Class']
X1=df_wine.drop(['Class'],axis = 1)

#Normalize the data to 0-10
X=((X1-X1.min())/(X1.max()-X1.min()))*10
print(X.head())
#Split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
print("=================================================================")
###################EDA####################

##Pairplot
#sns.set(style='whitegrid', context='notebook')
#sns.pairplot(df_wine[df_wine.columns], size=1.9)
#plt.show()
#print("=================================================================")
##Heatmap
#cm = np.corrcoef(df_wine[df_wine.columns].values.T)
#sns.set(font_scale=1.5)
#fig, hm = plt.subplots(figsize=(15,15))   
#hm = sns.heatmap(cm, cbar=True,annot=True, square=True,fmt='.2f',annot_kws={'size': 15},
#                 yticklabels=df_wine.columns,xticklabels=df_wine.columns)
#plt.figure(figsize=(50,50))
#plt.show()
print("=================================================================")
######Part 2: Logistic regression v. SVM - baseline############
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr = LogisticRegression(C=1000.0, random_state=1)
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
print('LR Train Score: ', metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.confusion_matrix(y_train, y_train_pred) )

y_test_pred = lr.predict(X_test)
print('LR Test Score: ', metrics.accuracy_score(y_test, y_test_pred) )
print( metrics.confusion_matrix(y_test, y_test_pred) )

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train,y_train)

y_train_pred = svm.predict(X_train)
print('SVM Train Score: ', metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.confusion_matrix(y_train, y_train_pred) )

y_test_pred = svm.predict(X_test)
print('SVM Test Score: ', metrics.accuracy_score(y_test, y_test_pred) )
print( metrics.confusion_matrix(y_test, y_test_pred) )
print("=================================================================")
####################plot_decision_regions########################
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)           
 
#########Part 3: PCA###################
from sklearn.decomposition import PCA
pca =  PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#LR
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca,y_train, classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Region of LR after PCA')
plt.legend(loc='upper right')
plt.show()
y_train_pred = lr.predict(X_train_pca)
print('LR Train Score: ', metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.confusion_matrix(y_train, y_train_pred) )
y_test_pred = lr.predict(X_test_pca)
print('LR Test Score: ', metrics.accuracy_score(y_test, y_test_pred) )
print( metrics.confusion_matrix(y_test, y_test_pred) )

#SVM
svm.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train, classifier = svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Region of SVM after PCA')
plt.legend(loc='upper right')
plt.show()
y_train_pred = svm.predict(X_train_pca)
print('SVM Train Score: ', metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.confusion_matrix(y_train, y_train_pred) )
y_test_pred = svm.predict(X_test_pca)
print('SVM Test Score: ', metrics.accuracy_score(y_test, y_test_pred) )
print( metrics.confusion_matrix(y_test, y_test_pred) )
print("=================================================================")

#########Part 4: LDA###################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

#LR
lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda,y_train, classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Region of LR after LDA')
plt.legend(loc='upper right')
plt.show()
y_train_pred = lr.predict(X_train_lda)
print('LR Train Score: ', metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.confusion_matrix(y_train, y_train_pred) )
y_test_pred = lr.predict(X_test_lda)
print('LR Test Score: ', metrics.accuracy_score(y_test, y_test_pred) )
print( metrics.confusion_matrix(y_test, y_test_pred) )

#SVM
svm.fit(X_train_lda,y_train)
plot_decision_regions(X_train_lda,y_train, classifier = svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Region of SVM after LDA')
plt.legend(loc='upper right')
plt.show()
y_train_pred = svm.predict(X_train_lda)
print('SVM Train Score: ', metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.confusion_matrix(y_train, y_train_pred) )
y_test_pred = svm.predict(X_test_lda)
print('SVM Test Score: ', metrics.accuracy_score(y_test, y_test_pred) )
print( metrics.confusion_matrix(y_test, y_test_pred) )
print("=================================================================")

#########Part 4: kPCA###################
from sklearn.decomposition import KernelPCA
kpca =  KernelPCA(n_components = 2, kernel='rbf',gamma=0.01)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)

#LR
lr.fit(X_train_kpca, y_train)
plot_decision_regions(X_train_kpca,y_train, classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Region of LR after kPCA')
plt.legend(loc='upper right')
plt.show()
y_train_pred = lr.predict(X_train_kpca)
print('LR Train Score: ', metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.confusion_matrix(y_train, y_train_pred) )
y_test_pred = lr.predict(X_test_kpca)
print('LR Test Score: ', metrics.accuracy_score(y_test, y_test_pred) )
print( metrics.confusion_matrix(y_test, y_test_pred) )

#SVM
svm.fit(X_train_kpca,y_train)
plot_decision_regions(X_train_kpca,y_train, classifier = svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Decision Region of SVM after kPCA')
plt.legend(loc='upper right')
plt.show()
y_train_pred = svm.predict(X_train_kpca)
print('SVM Train Score: ', metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.confusion_matrix(y_train, y_train_pred) )
y_test_pred = svm.predict(X_test_kpca)
print('SVM Test Score: ', metrics.accuracy_score(y_test, y_test_pred) )
print( metrics.confusion_matrix(y_test, y_test_pred) )
print("=================================================================")
########################################################
print("My name is Ning Fan")
print("My NetID is: ningfan2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")










