import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 目标类别数据
k = 5
kf = StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
for train_index,test_index in kf.split(X,y):
    unique_elements, counts = np.unique(y[test_index], return_counts=True)
    print("唯一元素:", unique_elements)
    print("对应数量:", counts)

# accuracies=[]
# for train_index,test_index in kf.split(X):
#     X_train,X_test = X[train_index],X[test_index]
#     y_train,y_test = y[train_index],y[test_index]
#     model = DecisionTreeClassifier()
#     model.fit(X_train,y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test,y_pred)
#     accuracies.append(accuracy)
# average_accuracy = np.mean(accuracies)
# print("平均准确率:", average_accuracy)


def hold_out(*arrays,test_size=None,train_size=None,random_state=None,shuffle=True,stratify=None):

    x_train, x_test, y_train, y_test = train_test_split(*arrays,test_size=test_size,train_size=train_size,random_state=random_state,shuffle=shuffle,stratify=stratify)
    return x_train, x_test, y_train, y_test


X_train, X_test, y_train, y_test = hold_out(X,y,test_size=0.3,random_state=42,stratify=y)


def kfold_cross_validation(*arrays,k,shuffle=True,random_state=None,stratify=None):
    if stratify == True:
        result_list = []
        skf = StratifiedKFold(n_splits=k,shuffle=shuffle,random_state=random_state)
        for train_index,test_index in skf.split(arrays[0],arrays[1]):
            x_train,x_test = arrays[0][train_index],arrays[0][test_index]
            y_train,y_test = arrays[1][train_index],arrays[1][test_index]
            sub_list = [x_train,x_test,y_train,y_test]
            result_list.append(sub_list)
        return result_list
    else:
        result_list = []
        kf = KFold(n_splits=k,shuffle=shuffle,random_state=random_state)
        for train_index,test_index in kf.split(arrays[0]):
            x_train,x_test = arrays[0][train_index],arrays[0][test_index]
            y_train,y_test = arrays[1][train_index],arrays[1][test_index]
            sub_list = [x_train,y_train,x_test,y_test]
            result_list.append(sub_list)
        return result_list
def leave_one_out(*array):
    result_list = []
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sub_list = [X_train,y_train,X_test,y_test]
        result_list.append(sub_list)
    return result_list
# result_list = kfold_cross_validation(X,y,k=5,random_state=42,stratify=True)


def bootStrap(*arrays,m,random_state=None):
    
    index = np.random.randint(0,m)
    train_indices = []
    while len(train_indices)< m:
        index = np.random.randint(0,m)
        train_indices.append(index)
        #用于从指定的整数区间(数据集索引范围)内随机抽取一个整数
    x_train = arrays[0][train_indices]
    y_train = arrays[1][train_indices]
    print(len(train_indices))


    test_indices = [i for i in range(m) if i not in train_indices]
    x_test = arrays[0][test_indices]
    y_test = arrays[1][test_indices]
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = bootStrap(X,y,m=X.shape[0],random_state=42)
print(y_train.shape)
print(X.shape[0])