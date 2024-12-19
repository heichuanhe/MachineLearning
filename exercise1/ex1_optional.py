import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import LinearRegressionModel as lrm
import FeatureScalingImplement as fs
from matplotlib.widgets import Slider
from matplotlib import cm
import itertools
import DataPartitioning as dp
#数据处理
datafile = 'data/ex1data2.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
x = np.transpose(cols[:-1])
y = np.transpose(cols[-1:])
#数据初始化
standard_arrays_nor = fs.standardNormalization(x,y)
x_standard = standard_arrays_nor[0]
y_standard = standard_arrays_nor[1]
mean_arrays_nor = fs.meanNormalization(x,y)
x_mean = mean_arrays_nor[0]
y_mean = mean_arrays_nor[1]
z_score_arrays_nor = fs.z_scoreNormalization(x,y)
x_zScore = z_score_arrays_nor[0]
y_zScore = z_score_arrays_nor[1]
x_standard = np.insert(x_standard,0,1,axis=1)
x_mean = np.insert(x_mean,0,1,axis=1)
x_zScore = np.insert(x_zScore,0,1,axis=1)
#数据分割  仅用留出法
x_standard_train, x_standard_test, y_standard_train, y_standard_test = dp.hold_out(x_standard,y_standard,test_size=0.3,random_state=42)
x_mean_train, x_mean_test, y_mean_train, y_mean_test = dp.hold_out(x_mean,y_mean,test_size=0.3,random_state=42)
x_zScore_train, x_zScore_test, y_zScore_train, y_zScore_test = dp.hold_out(x_zScore,y_zScore,test_size=0.3,random_state=42)

#设定超参数
iterations = 5000
alpha = 0.05
init_theta_standard = np.zeros((x_standard_train.shape[1],1))
init_theta_mean = np.zeros((x_mean_train.shape[1],1))
init_theta_zScore = np.zeros((x_zScore_train.shape[1],1))

#训练模型
standard_theta = lrm.model_fit(alpha=alpha,x=x_standard_train,y=y_standard_train,theta_start=init_theta_standard,iterations=iterations)
mean_theta = lrm.model_fit(alpha=alpha,x=x_mean_train,y=y_mean_train,theta_start=init_theta_mean,iterations=iterations)
zScore_theta = lrm.model_fit(alpha=alpha,x=x_zScore_train,y=y_zScore_train,theta_start=init_theta_zScore,iterations=iterations)

#预测
y_standard_train_pred = lrm.model_predict(theta=standard_theta,x=x_standard_train)
y_standard_test_pred = lrm.model_predict(theta=standard_theta,x=x_standard_test)

y_mean_train_pred = lrm.model_predict(theta=mean_theta,x=x_mean_train)
y_mean_test_pred = lrm.model_predict(theta=mean_theta,x=x_mean_test)

y_zScore_train_pred = lrm.model_predict(theta=zScore_theta,x=x_zScore_train)
y_zScore_test_pred = lrm.model_predict(theta=zScore_theta,x=x_zScore_test)

#模型评估
mse_standard_train = lrm.model_mean_squared_error(y_train=y_standard_train,y_pred=y_standard_train_pred)
mse_standard_test = lrm.model_mean_squared_error(y_train=y_standard_test,y_pred=y_standard_test_pred)

mse_mean_train = lrm.model_mean_squared_error(y_train = y_mean_train,y_pred=y_mean_train_pred)
mse_mean_test = lrm.model_mean_squared_error(y_train = y_mean_test,y_pred=y_mean_test_pred)

mse_zScore_train = lrm.model_mean_squared_error(y_train = y_zScore_train,y_pred=y_zScore_train_pred)
mse_zScore_test = lrm.model_mean_squared_error(y_train = y_zScore_test,y_pred=y_zScore_test_pred)


# print(x_standard)
# print("==============================================================>")
# print(x_mean)
# print("==============================================================>")
# print(x_zScore)


print("standard_Jtrain:",mse_standard_train)
print("standard_Jtest:",mse_standard_test)
print("mean_Jtrain:",mse_mean_train)
print("mean_Jtest:",mse_mean_test)
print("zScore_Jtrain:",mse_zScore_train)
print("zScore_Jtest:",mse_zScore_test)

def plotScatter(x,y):
    fig,a = plt.subplots(nrows=1,ncols=2)
    fig.set_size_inches(10,6)
    a[0].plot(x[:,1],x[:,2],'rx',markersize=10)
    a[0].grid(True) #Always plot.grid true!
    a[0].set_ylabel('bedrooms')
    a[0].set_xlabel('size')
    # min_x_value, max_x_value = 0, 4500  # 这里只是示例值，你需要根据实际数据修改
    # min_y_value, max_y_value = 0, 100   # 这里只是示例值，你需要根据实际数据修改
    # a[0].set_xlim([min_x_value, max_x_value])
    # a[0].set_ylim([min_y_value, max_y_value])
def plotConvergence(jvec):
    plt.figure(figsize=(10,6))
    plt.plot(range(len(jvec)),jvec,'bo')
    plt.grid(True)
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration number")
    plt.ylabel("Cost function")
    dummy = plt.xlim([-0.05*iterations,1.05*iterations])

def plot3DScatter(x,y,theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,1], x[:,2], y[:,0])
    x1_grid, x2_grid = np.meshgrid(np.linspace(min(x[:,1]), max(x[:,1]), 10), np.linspace(min(x[:,2]), max(x[:,2]), 10))
    y_grid = theta[0] + theta[1] * x1_grid + theta[2] * x2_grid
    ax.plot_surface(x1_grid, x2_grid, y_grid, alpha = 0.3)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

def plotSurface(x,y,theta):
    

    pass

plot3DScatter(x_standard_train,y_standard_train,standard_theta)
plt.show()

