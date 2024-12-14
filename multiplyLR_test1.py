import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
"""
数据预处理
"""
datafile = 'data/ex1data2.txt'
cols = np.loadtxt(datafile,delimiter=',',unpack=True)
x = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size
x = np.insert(x,0,1,axis=1)
iterations = 1500  #迭代次数
alpha = 0.01  #学习率/步幅
initial_theta = np.zeros((x.shape[1],1))

def hypothesis(theta,x):
    """
    m*3  *  3*1  = m*1
    theta = [[theta[0]],
             [theta[1]],
             [theta[2]]]
             
             """
    
    #h(x) = theta[0]+theta[1]*x1+theta[2]*x2
    return np.dot(x,theta)

def computeCost(mytheta,x,y):
    dot_result = np.dot((hypothesis(mytheta,x)-y).T,(hypothesis(mytheta,x)-y))
    if dot_result.size==1:
        single_element = dot_result[0,0]
        return float((1./2*m))*single_element
    else:
        raise ValueError("Unexpected result shape from np.dot operation")

def descendGradient(x,y,theta_start):
    theta = theta_start
    jvec = [] #Used to plot cost as function of iteration
    thetahistory = [] #Used to visualize the minimization path later on
    for i in range(iterations):
        tmptheta = theta
        cost = computeCost(tmptheta,x,y)
        jvec.append(cost)
        thetahistory.append(list(theta[:,0]))
        for j in range(len(theta)):
            tmptheta[j] = tmptheta[j] - (alpha/m)*np.sum((hypothesis(tmptheta,x) - y)*np.array(x[:,j]).reshape(m,1))
        theta = tmptheta
    return theta ,thetahistory,jvec

fig,a = plt.subplot(1,2)
plt.subplots_adjust(bottom=0.2)
fig.set_size_inches(10,6)
