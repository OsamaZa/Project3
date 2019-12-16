import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import collections
from sklearn.model_selection import KFold

np.random.seed(10)

df_BC = pd.read_csv("breast.txt",sep=",")
df_BC=df_BC.drop(['id'], axis=1)

X = df_BC.loc[:, df_BC.columns != 'Class'].values
y = df_BC.loc[:, df_BC.columns == 'Class'].values

      
onehotencoder = OneHotEncoder(categories="auto")
X = ColumnTransformer([("",onehotencoder,[1,2,3,4,5,6,7,8]),],remainder="passthrough").fit_transform(X)
y = np.ravel(y)

plt.figure()
correlation=df_BC.corr()
ax = sns.heatmap(correlation)
plt.title("Heatmap of the correlation matrix")

plt.figure()
corre=correlation["Class"]
corre.drop(corre.index[-1]).plot.barh()
plt.title("Correlation values for 'Class' aka negativ/positive")

print("_____Data distribution_____")
print(collections.Counter(y))

def MLP_Class(nodes,active,solv,l2,learn_r,X,y):
    kf = KFold(n_splits=5,shuffle=True)
    kf.get_n_splits(X)
    PRE=[]
    Confusion=[]
    y_prediction=[]
    Distri=[]
    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X[train_index], X[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]
        BC = MLPClassifier(hidden_layer_sizes=(nodes,), activation=active,
                           solver=solv, alpha=l2,learning_rate=learn_r)
        BC.fit(X_train_kf,y_train_kf)
        y_pred = BC.predict(X_test_kf)
        PRE.append(accuracy_score(y_test_kf,y_pred))
        Confusion.append(confusion_matrix(y_test_kf, y_pred))
        y_prediction.append(y_pred)
        Distri.append(collections.Counter(y_pred))
    PRECISION.append(np.mean(PRE))
    Confusion_matrix.append([Confusion])
    MLP_model.append([nodes,active,solv,l2,learn_r])
    Predictions.append([y_prediction])
    Distribution.append(Distri)

Hidden_nodes = (25,75,100)
Activ_func = ("identity","logistic","tanh","relu")
Solver = ("lbfgs","sgd","adam")
Alpha_l2 = np.logspace(-8,-1,5)
Learning_rate = ("constant","invscaling","adaptive")

Predictions = []
PRECISION = []
Confusion_matrix = []
MLP_model = []
Distribution=[]

for nodes in Hidden_nodes:
    for active in Activ_func:
        for solv in Solver:
            for l2 in Alpha_l2:
                for learn_r in Learning_rate:
                    MLP_Class(nodes,active,solv,l2,learn_r,X,y)                        

from more_itertools import locate
maxpre = np.argmax(np.unique(PRECISION))
maxprecision_indexes = list(locate(PRECISION, lambda a: a == np.unique(PRECISION)[maxpre]))
print("_____The models with highest accuracy_____")
for index in maxprecision_indexes:
    print("_____The model_____")
    print(MLP_model[index])
    print("_____Accuracy Score for test data_____")
    print(PRECISION[index])
    print("_____Confusion Matrix_____")
    print(Confusion_matrix[index])
    print("_____Distribution of predictions_____")
    print(Distribution[index])

plt.show()

