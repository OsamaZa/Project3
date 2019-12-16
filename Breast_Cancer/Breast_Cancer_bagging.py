import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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

cost = "gini" # Splitting criterion
k = 5 # Number of subsets
kf = KFold(n_splits=k,shuffle=True) # For indices of the subsets
l_estimators = np.array([10,50,100,200,500,1000]) # List with the number of base estimators to be combined
l_maxsamples = np.array([10,100,200,300,400,500]) # List with max samples drawn to train each decision tree
values = np.zeros((len(l_estimators),len(l_maxsamples))) # Making matrix to store the accuracy scores
confusion_matrices = np.zeros((len(l_estimators),len(l_maxsamples),2,2)) # Making matrix to the confusion matrices
for l, i in zip(range(len(l_estimators)), l_estimators):
    for m, j in zip(range(len(l_maxsamples)), l_maxsamples):
        # Setting up bagging classifier
        clf = BaggingClassifier(
                DecisionTreeClassifier(criterion=cost), n_estimators=i,
            max_samples=j)
        accuracy_sum = 0 # Sum of the accuracy scores for each cross-validation
        confusion_sum = np.zeros((2,2)) # Sum of the confusion matrices for each cross validation
        # K-fold cross-validation
        for train, test in kf.split(X):
            clf.fit(X[train], y[train]) # Training the model
            yPred = clf.predict(X[test]) # Using the trained tree to predict values for the test data
            accuracy_sum += accuracy_score(y[test],yPred) # Adding the accuracy score to the sum of them
            confusion_sum += confusion_matrix(y[test],yPred) # Adding the confusion matrix to the sum of them
        values[l,m] = accuracy_sum/k # Storing the mean accuracy score
        confusion_matrices[l,m,:,:] = confusion_sum/k # Storing the mean confusion matrix

# Selecting the best model and its accuracy and confusion matrix
best_index = np.where(values == np.max(values))
best_accuracy = float(values[best_index[0],best_index[1]])
best_confusion = confusion_matrices[best_index[0],best_index[1],:,:]
best_estimator = int(l_estimators[best_index[0]])
best_maxsample = int(l_maxsamples[best_index[1]])
print("Best accuracy for decision tree with bagging: {:.6f} for %s estimators and %s samples drawn for each estimator".format(best_accuracy) %(best_estimator,best_maxsample))
print(best_confusion)
# Plotting the accuracy scores for the different parameters
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
res = ax.imshow(values, cmap=plt.cm.jet, interpolation="nearest")

height, width = values.shape

for x in range(width):
    for y in range(height):
        ax.annotate(round(values[y][x], 6), xy=(x, y), 
                    horizontalalignment='center',
                    verticalalignment='center')
cb = fig.colorbar(res)
plt.xticks(range(width), l_maxsamples[:width])
plt.yticks(range(height), l_estimators[:height])
plt.title('accuracy scores with %s index as the splitting criterion' %cost)
plt.xlabel('maximum number of samples drawn to train each decision tree')
plt.ylabel('number of decision trees combined')
plt.show()
