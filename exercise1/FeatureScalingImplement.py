import numpy as np
from sklearn.preprocessing import StandardScaler


x = np.array([[1,2],[2,5],[3,7],[4,8],[5,15]])
y = np.arange(1,6)

def standardNormalization(*arrays):
    result_arrays = []
    for array in arrays:
        array_nor = np.zeros(array.shape)
        if len(array.shape) == 1:
            column_max = np.max(array)
            result_arrays.append(array/float(column_max))
        else:
            for i in range(array.shape[1]):
                array_nor[:,i] = array[:,i]/float(np.max(array[:,i]))
            result_arrays.append(array_nor)
    return result_arrays


def meanNormalization(*arrays):
    result_arrays = []
    for array in arrays:
        array_nor = np.zeros(array.shape)
        if len(array.shape) == 1:
            mean_value = np.mean(array)
            column_max = np.max(array)
            column_min = np.min(array)
            result_arrays.append((array-mean_value)/float(column_max-column_min))
        else:
            mean_value_per_column = np.mean(array,axis=0)
            for i in range(array.shape[1]):
                mean_value = mean_value_per_column[i]
                array_nor[:,i] = (array[:,i]-mean_value)/float(np.max(array[:,i])-np.min(array[:,i]))
            result_arrays.append(array_nor)
    return result_arrays



def z_scoreNormalization(*arrays):
    result_arrays = []
    for array in arrays:
        array_nor = np.zeros(array.shape)
        if len(array.shape) == 1:
            mean_value = np.mean(array)
            std_value = np.std(array)
            result_arrays.append((array-mean_value)/float(std_value))
        else:
            mean_value_per_column = np.mean(array,axis=0)
            std_value_per_column = np.std(array,axis=0)
            for i in range(array.shape[1]):
                mean_value = mean_value_per_column[i]
                std_value = std_value_per_column[i]
                array_nor[:,i] = (array[:,i]-mean_value)/std_value
            result_arrays.append(array_nor)
    return result_arrays

def z_score_normalization_sklearn(arr):
    scaler = StandardScaler()
    if len(arr.shape) == 1:
        arr_reshaped = arr.reshape(-1, 1)
        scaler.fit(arr_reshaped)
        return scaler.transform(arr_reshaped).ravel()
    else:
        return scaler.fit_transform(arr)
