import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import LinearRegressionModel as LRM
import FeatureScalingImplement as fs
from matplotlib.widgets import Slider
from matplotlib import cm
import itertools
#数据处理
datafile = 'data/ex1data2.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
x = np.transpose(cols[:-1])
y = np.transpose(cols[-1:])
arrays_nor = fs.standardNormalization(x,y)
x_standard = arrays_nor[0]
y_standard = arrays_nor[1]


# x_standard = fs.z_score_normalization_sklearn(x)
# y_standard = fs.z_score_normalization_sklearn(y)
x_standard = np.insert(x_standard,0,1,axis=1)
print(x_standard)
print('==========================================')
print(y_standard)
def plotScatter(x,y):
    fig,a = plt.subplots(nrows=1,ncols=2)
    fig.set_size_inches(10,6)
    a[0].plot(x_standard[:,1],x_standard[:,2],'rx',markersize=10)
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

def plot3DScatter(x,y):
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
iterations = 5000
alpha = 0.05
initial_theta = np.zeros((x_standard.shape[1],1))

theta, thetahistory, jvec = LRM.descentGradient(alpha,x_standard,y_standard,initial_theta,iterations)
last_cost = LRM.computeCost(x_standard,y_standard,theta)
print(jvec[-1])
print(last_cost)
plotConvergence(jvec)
plot3DScatter(x_standard,y_standard)
plt.show()


