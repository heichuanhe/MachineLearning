import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 目标类别数据
# 创建决策树分类器模型


k = 5
kf = KFold(n_splits=k,shuffle=True,random_state=42)

for train_index,test_index in kf.split(X):
    print("train_index",train_index)
    print("test_index",test_index)

accuracies=[]
for train_index,test_index in kf.split(X):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracies.append(accuracy)
average_accuracy = np.mean(accuracies)
# print("平均准确率:", average_accuracy)


def hold_out(*arrays,test_size=None,train_size=None,random_state=None,shuffle=True,stratify=None):

    x_train, x_test, y_train, y_test = train_test_split(*arrays,test_size=test_size,train_size=train_size,random_state=random_state,shuffle=shuffle,stratify=stratify)
    return x_train, x_test, y_train, y_test


X_train, X_test, y_train, y_test = hold_out(X,y,test_size=0.3,random_state=42,stratify=y)


def kfold_cross_validation():

    pass