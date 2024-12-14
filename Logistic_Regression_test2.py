import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy.special import expit

datafile = 'data/ex2data2.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
x = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size
x = np.insert(x,0,1,axis=1)

pos = np.array([x[i] for i in range(m) if y[i] == 1])
neg = np.array([x[i] for i in range(m) if y[i] == 0])
def plotData():
    plt.figure(figsize=(6,6))
    plt.plot(pos[:,1],pos[:,2],'k+',label='y=1')
    plt.plot(neg[:,1],neg[:,2],'yo',label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.grid(True)

def plotSigmoid():
    myx = np.arange(-10,10,.1)
    plt.plot(myx,expit(myx))
    plt.title('Woohoo this looks like a sigmoid function to me')
    plt.grid(True)
    plt.show()
def h(myx,mytheta):
    return expit(np.dot(myx,mytheta))

def computeCost(mytheta,myx,myy,mylambda = 0.):
    """
    J(w,b) = -1/mΣ(m,i=1)[y*log(f(x))+(1-y)log(1-f(x))] + λ/2mΣ(n,j=1)wj2
    J(w,b) = 1/mΣ(m,i=1)[-y*log(f(x))-(1-y)log(1-f(x))+λ/2Σ(n,j=1)wj2]
    """
    term1 = np.dot(-y.T,np.log(h(myx,mytheta)))
    term2 = np.dot(1-y.T,np.log(1-h(myx,mytheta)))
    regterm = (mylambda/2)*np.sum(np.dot(mytheta[1:].T,mytheta[1:]))
    return float((1./m)*(np.sum(term1 - term2 )+regterm))



# def optimizeTheta(mytheta,myx,myy,mylambda=0):
#     result = optimize.fmin(computeCost,x0=mytheta,args=(myx,myy,mylambda),maxiter=400,full_output=True)
#     return result[0],result[1]

def mapFeature(x1col,x2col):
    degrees = 6
    out = np.ones((x1col.shape[0],1))
    for i in range(1,degrees+1):
        for j in range(0,i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term = (term1*term2).reshape(term1.shape[0],1)
            out = np.hstack((out,term))
    return out
#x1  x2  最高6次幂  
mappedX = mapFeature(x[:,1],x[:,2])
print(mappedX.shape) 
initial_theta_mapped = np.zeros((mappedX.shape[1],1))
def optimizeRegularizedTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.minimize(computeCost, mytheta, args=(myX, myy, mylambda),  method='BFGS', options={"maxiter":500, "disp":False} )
    return np.array([result.x]), result.fun 
theta, mincost = optimizeRegularizedTheta(initial_theta_mapped.reshape(mappedX.shape[1],),mappedX,y)
print(theta)
print('cost:',mincost)

def plotBoundary(mytheta, myX, myy, mylambda=0.):
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    theta, mincost = optimizeRegularizedTheta(mytheta.reshape(myX.shape[1],),myX,myy,mylambda)
    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta,myfeaturesij.T)
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")
plt.figure(figsize=(12,10))
plt.subplot(221)
plotData()
plotBoundary(theta,mappedX,y,100.)
plt.show()

# plt.subplot(222)
# plotData()
# plotBoundary(theta,mappedX,y,1.)

# plt.subplot(223)
# plotData()
# plotBoundary(theta,mappedX,y,10.)

# plt.subplot(224)
# plotData()
# plotBoundary(theta,mappedX,y,100.)




