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
print(x[:,0])
print(x[:,1])
print(y[:,0])
print(y.shape)
def plot3DScatter(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(x[:,0], x[:,1], y[:,0])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    points = scat.get_offsets()
    num_points = len(points)
    print(f"散点图中的点数为: {num_points}")

plot3DScatter(x,y)
plt.show()