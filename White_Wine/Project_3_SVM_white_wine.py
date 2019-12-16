import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import collections
from sklearn.model_selection import KFold

np.random.seed(10)

# White wine
df_w = pd.read_excel("winequality_white.xls")

plt.figure()
corr = df_w.corr()
ax = sns.heatmap(corr)
plt.title("Heatmap of the correlation matrix white wine")

plt.figure()
cor=corr["quality"]
cor.drop(cor.index[-1]).plot.barh()
plt.title("Correlation values for 'quality'")

X = df_w.loc[:, df_w.columns[:11]].values
y = df_w.loc[:, df_w.columns[11]].values

sc = StandardScaler()
X[:] = sc.fit_transform(X[:])

print("_____Data distribution_____")
print(collections.Counter(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85)

print("_____Data distribution for test and train data_____")
print(collections.Counter(y_train),len(y_train))


def SVM(c,kernel,degree,gamma,X_train,X_test,y_train,y_test):
    kf = KFold(n_splits=5,shuffle=True)
    kf.get_n_splits(X_train)
    PRE=[]
    Confusion=[]
    y_prediction=[]
    Distri=[]
    for train_index, test_index in kf.split(X_train):
        X_train_kf, X_test_kf = X_train[train_index], X_train[test_index]
        y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]
        BC = SVC(C=c, kernel=kernel, degree=degree, gamma=gamma)
        BC.fit(X_train_kf,y_train_kf)
        y_pred = BC.predict(X_test_kf)
        Distri.append(collections.Counter(y_pred))
        PRE.append(accuracy_score(y_test_kf,y_pred))
        Confusion.append(confusion_matrix(y_test_kf, y_pred))
        y_prediction.append(y_pred)
    PRECISION.append(np.mean(PRE))
    Confusion_matrix.append([Confusion])
    SVM_model.append([c,kernel,degree,gamma])
    Predictions.append([y_prediction])
    Distribution.append(Distri)
    
C = np.logspace(-3,5,8)
Kernel = ("poly","rbf","sigmoid")
Degree = (0,3,6,9,12,15,18)
Gamma = ("scale","auto")

Predictions = []
PRECISION = []
Confusion_matrix = []
SVM_model = []
Distribution=[]

for c in C:
    for kernel in Kernel:
        if kernel == "poly":
            for degree in Degree:
                for gamma in Gamma:
                    SVM(c,kernel,degree,gamma,X_train,X_test,y_train,y_test)
        else:
            for gamma in Gamma:
                SVM(c,kernel,degree,gamma,X_train,X_test,y_train,y_test)

from more_itertools import locate
maxpre = np.argmax(np.unique(PRECISION))
maxprecision_indexes = list(locate(PRECISION, lambda a: a == np.unique(PRECISION)[maxpre]))
print("_____The models with highest accuracy_____")
for index in maxprecision_indexes:
    print("_____The model_____")
    print(SVM_model[index])
    print("_____Accuracy Score for test data_____")
    print(PRECISION[index])
    print("_____Confusion Matrix_____")
    print(Confusion_matrix[index])
    print("_____Distribution of predictions_____")
    print(Distribution[index])

plt.show()

