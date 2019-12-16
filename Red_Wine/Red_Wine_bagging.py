import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

# Setting the seed
np.random.seed(10)

# Reads the data
df = pd.read_excel("winequality_red.xls")

# Heatmap if the correlation matrix
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

cost = "gini" # Splitting criterion
k = 5 # Number of subsets
kf = KFold(n_splits=k,shuffle=True,random_state=10) # For indices of the subsets
l_estimators = np.array([10,50,100,200,500,1000]) # List with the number of base estimators to be combined
l_maxsamples = np.array([500,700,1000,1200]) # List with max samples drawn to train each decision tree
values = np.zeros((len(l_estimators),len(l_maxsamples))) # Making matrix to store the accuracy scores
confusion_matrices = np.zeros((len(l_estimators),len(l_maxsamples),6,6)) # Making matrix to the confusion matrices
for l, i in zip(range(len(l_estimators)), l_estimators):
    for m, j in zip(range(len(l_maxsamples)), l_maxsamples):
        # Setting up bagging classifier
        clf = BaggingClassifier(
                DecisionTreeClassifier(criterion=cost), n_estimators=i,
            max_samples=j)
        accuracy_sum = 0 # Sum of the accuracy scores for each cross-validation
        confusion_sum = np.zeros((6,6)) # Sum of the confusion matrices for each cross validation
        # K-fold cross-validation
        for train, test in kf.split(X):
            clf.fit(X[train], y[train]) # Training the model
            yPred = clf.predict(X[test]) # Using the trained tree to predict values for the test data
            accuracy_sum += accuracy_score(y[test],yPred) # Adding the accuracy score to the sum of them
            k_confusion = confusion_matrix(y[test],yPred) # The confusion matrix
            # Making the confusion matrix a 6x6 matrix if it is 5x5
            if k_confusion.shape[0] == 5:
                M = np.zeros((6,6))
                M[1:6,1:6] = k_confusion
                k_confusion = M
            confusion_sum += k_confusion # Adding the confusion matrix to the sum of them
        values[l,m] = accuracy_sum/k # Storing the mean accuracy score
        confusion_matrices[l,m,:,:] = confusion_sum/k # Storing the mean confusion matrix

# Selecting the best model and its accuracy and confusion matrix
best_index = np.where(values == np.max(values))
best_accuracy = float(values[best_index[0],best_index[1]])
best_confusion = confusion_matrices[best_index[0],best_index[1],:,:]
best_estimator = int(l_estimators[best_index[0]])
best_maxsample = int(l_maxsamples[best_index[1]])
np.set_printoptions(formatter={'float_kind':'{:f}'.format}) # To get float numbers in the average confusion matrix
print("Best accuracy for decision tree with bagging: {:.6f} for %s estimators and %s samples drawn for each estimator".format(best_accuracy) %(best_estimator,best_maxsample))
print(best_confusion)
print(Counter(y))
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

