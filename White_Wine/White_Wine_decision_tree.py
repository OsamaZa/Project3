import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from sklearn.model_selection import KFold
# Setting the seed
np.random.seed(10)

# Reads the data
df = pd.read_excel("winequality_white.xls")

# Heatmap of the correlation matrix
plt.figure()
corr = df.corr()
ax = sns.heatmap(corr)
plt.title("Heatmap of the correlation matrix red wine")

# Correlation of quality
plt.figure()
cor=corr["quality"]
cor.drop(cor.index[-1]).plot.barh()
plt.title("Correlation values for 'quality'")

# Features and target
X = df.loc[:, df.columns[:11]].values
y = df.loc[:, df.columns[11]].values

# Scaling
sc = StandardScaler()
X[:] = sc.fit_transform(X)

# Train-test split
trainingShare = 0.8
XTrain, XTest, yTrain, yTest=train_test_split(X, y, train_size=trainingShare, \
                                              test_size = 1-trainingShare)

clf = DecisionTreeClassifier(criterion="entropy") # Selecting classifier model
k = 5 # Number of subsets
kf = KFold(n_splits=k,shuffle=True) # For indices of the subsets
accuracy_sum = 0 # Sum of the accuracy scores for each cross-validation
confusion_sum = np.zeros((7,7)) # Sum of the confusion matrices for each cross validation
# K-fold cross-validation
for train, test in kf.split(X):
    clf.fit(X[train], y[train]) # Training the model
    yPred = clf.predict(X[test]) # Predicting target values
    accuracy_sum += accuracy_score(y[test],yPred) # Adding the accuracy score to the sum of them
    k_confusion = confusion_matrix(y[test],yPred) # The confusion matrix
    # Making the confusion matrix a 7x7 matrix if it is 6x6
    if k_confusion.shape[0] == 6:
        M = np.zeros((7,7))
        M[0:6,0:6] = k_confusion
        k_confusion = M
    confusion_sum += k_confusion # Adding the confusion matrix to the sum of them

accuracy = accuracy_sum/k # The mean of the accuracy scores
confusion = confusion_sum/k # The mean of the confusion matrices
np.set_printoptions(formatter={'float_kind':'{:f}'.format}) # To get float numbers in the average confusion matrix
print("Test set accuracy with decision trees: {:.6f}".format(accuracy))
print(confusion)
print(Counter(y))
