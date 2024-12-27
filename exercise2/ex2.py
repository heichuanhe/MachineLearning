import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit
from scipy import optimize
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

datafile = 'data/ex2data1.txt'
#!head $datafile
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True) #Read in comma separated data
##Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size # number of training examples
##Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)

X = standardNormalization(X)[0]


pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

def plotData():
    plt.figure(figsize=(10,6))
    plt.plot(pos[:,1],pos[:,2],'k+',label='Admitted')
    plt.plot(neg[:,1],neg[:,2],'yo',label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.grid(True)
    
#This code I took from someone else (the OCTAVE equivalent was provided in the HW)
def mapFeature( x1col, x2col):
    """ 
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of featuers as described in the homework assignment
    """
    degrees = 4
    out = np.ones( (x1col.shape[0], 1) )
    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
            out   = np.hstack(( out, term ))
    return out
mappedX1 = mapFeature(x1col=X[:,1],x2col=X[:,2])
print(mappedX1.shape)
def h(mytheta,myX): #Logistic hypothesis function
    return expit(np.dot(myX,mytheta))
#Cost function, default lambda (regularization) 0
epsilon = 1e-10
def computeCost(mytheta,myX,myy,mylambda = 0.): 
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    Note this includes regularization, if you set mylambda to nonzero
    For the first part of the homework, the default 0. is used for mylambda
    """
    #note to self: *.shape is (rows, columns)
    term1 = np.dot(-np.array(myy).T,np.log(h(mytheta,myX)))
    # print("term1:",term1)
    term2 = np.dot((1-np.array(myy)).T,np.log(1-h(mytheta,myX)+epsilon))
    # print("term2:",term2)
    # print(term1-term2)            #结果是一个向量矩阵
    # print(np.sum(term1-term2))   #结果是一个标量
    regterm = (mylambda/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:])) #Skip theta0 ,对bias不进行正则化处理
    return float( (1./m) * ( np.sum(term1 - term2) + regterm ) )

initial_theta = np.zeros((mappedX1.shape[1],1))

options = {'maxiter': 5000, 'disp': False}
def optimizeTheta_new(mytheta,myx,myy,mylambda=0.):
    result = optimize.minimize(computeCost,x0=mytheta,args=(myx,myy,mylambda),method='BFGS',options=options)
    #return theta , mincost
    return np.array([result.x]),result.fun

theta,mincost= optimizeTheta_new(initial_theta.reshape(initial_theta.shape[0]),mappedX1,y)


def plotBoundary(mytheta, myX, myy, mylambda=0.):
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    theta,mincost= optimizeTheta_new(initial_theta.reshape(initial_theta.shape[0]),mappedX1,y,mylambda)
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

plotBoundary(theta,mappedX1,y,0.)

plt.subplot(222)

plotBoundary(theta,mappedX1,y,1.)

plt.subplot(223)

plotBoundary(theta,mappedX1,y,10.)

plt.subplot(224)

plotBoundary(theta,mappedX1,y,100.)
plt.show()