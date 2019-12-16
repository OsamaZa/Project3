import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
# Setting the seed
np.random.seed(10)

# Reads the data
cancer = pd.read_csv("breast-cancer-wisconsin.txt",sep=",",header=None)
# Removing rows with '?'
cancer = cancer[(cancer != "?").all(1)]
# Features and targets
X = cancer.loc[:,1:9]
y = cancer.loc[:,10]
y = np.ravel(y)
# Categorical variabler to one-hot's
onehotencoder = OneHotEncoder(categories="auto")
X = ColumnTransformer(
    [("", onehotencoder,[0,1,2,3,4,5,6,7,8]),],
    remainder="passthrough"
    ).fit_transform(X)

clf = DecisionTreeClassifier(criterion="gini") # Selecting classifier model
k = 5 # Number of subsets
kf = KFold(n_splits=k,shuffle=True) # For indices of the subsets
accuracy_sum = 0 # Sum of the accuracy scores for each cross-validation
confusion_sum = np.zeros((2,2)) # Sum of the confusion matrices for each cross validation
# K-fold cross-validation
for train, test in kf.split(X):
    clf.fit(X[train], y[train]) # Training the model
    yPred = clf.predict(X[test]) # Predicting target values
    accuracy_sum += accuracy_score(y[test],yPred) # Adding the accuracy score to the sum of them
    confusion_sum += confusion_matrix(y[test],yPred) # Adding the confusion matrix to the sum of them
accuracy = accuracy_sum/k # The mean of the accuracy scores
confusion = confusion_sum/k # The mean of the confusion matrices
print("Test set accuracy with decision trees: {:.6f}".format(accuracy))
print(confusion)

