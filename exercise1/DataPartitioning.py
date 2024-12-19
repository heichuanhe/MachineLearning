import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


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
    for train_index, test_index in loo.split(array[0]):
        X_train, X_test = array[0][train_index], array[0][test_index]
        y_train, y_test = array[1][train_index], array[1][test_index]
        sub_list = [X_train,y_train,X_test,y_test]
        result_list.append(sub_list)
    return result_list


#自助法
def bootStrap(*arrays,m,random_state=None):
    
    index = np.random.randint(0,m)
    train_indices = []
    while len(train_indices)< m:
        index = np.random.randint(0,m)
        train_indices.append(index)
        #用于从指定的整数区间(数据集索引范围)内随机抽取一个整数
    x_train = arrays[0][train_indices]
    y_train = arrays[1][train_indices]


    test_indices = [i for i in range(m) if i not in train_indices]
    x_test = arrays[0][test_indices]
    y_test = arrays[1][test_indices]
    return x_train, x_test, y_train, y_test
