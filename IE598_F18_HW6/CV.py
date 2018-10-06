import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)


############Decision Tree Classifier####################
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',max_depth=4, random_state=1)

#############Random test train splits###################
train_accuracy = np.empty(10)
test_accuracy = np.empty(10)
for i in range(1,10):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=i)
    tree.fit(X_train,y_train)
    train_accuracy[i] = tree.score(X_train,y_train)
    test_accuracy[i] = tree.score(X_test,y_test)
    
print('Random TTS train accuracy scores: %s' % train_accuracy)
print('Random TTS test accuracy scores: %s' % test_accuracy)
print('Random TTS train accuracy: %.3f +/- %.3f' % (np.mean(train_accuracy), np.std(train_accuracy)))
print('Random TTS test accuracy: %.3f +/- %.3f' % (np.mean(test_accuracy), np.std(test_accuracy)))
print('=================================================')
###########Cross Validation#######################
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=42)

tree.fit(X_train, y_train)

scores = cross_val_score(estimator = tree, X=X_train,y=y_train,cv=10,n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

print('Out-of-sample Accuracy Score: ', tree.score(X_test,y_test))

print('=================================================')


print("My name is Ning Fan")
print("My NetID is: 673869376")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################










