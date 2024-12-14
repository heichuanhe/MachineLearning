import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy.special import expit

datafile = 'data/ex2data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
x = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size
x = np.insert(x,0,1,axis=1)

pos = np.array([x[i] for i in range(m) if y[i] == 1])
neg = np.array([x[i] for i in range(m) if y[i] == 0])
def plotData():
    plt.figure(figsize=(10,6))
    plt.plot(pos[:,1],pos[:,2],'k+',label='Admitter')
    plt.plot(neg[:,1],neg[:,2],'yo',label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.grid(True)

# plotData()
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
    epsilon = 1e-10
    term1 = np.dot(-y.T,np.log(h(myx,mytheta)))
    term2 = np.dot(1-y.T,np.log(1-h(myx,mytheta)+epsilon))
    regterm = (mylambda/2)*np.sum(np.dot(mytheta[1:].T,mytheta[1:]))
    return float((1./m)*(np.sum(term1 - term2 )+regterm))


initial_theta = np.zeros((x.shape[1],1))
cost = computeCost(initial_theta,x,y)


def optimizeTheta(mytheta,myx,myy,mylambda=0):
    result = optimize.fmin(computeCost,x0=mytheta,args=(myx,myy,mylambda),maxiter=400,full_output=True)
    return result[0],result[1]

theta,mincost = optimizeTheta(initial_theta,x,y)
# print(theta)
# print(mincost)
iteration = 100000
alpha = 0.005
def descendGradient(x,y,theta_start):
    #梯度下降算法
    theta = theta_start
    jvec = [] #Used to plot cost as function of iteration
    thetahistory = [] #Used to visualize the minimization path later on
    for _ in range(iteration):
        tmptheta = theta
        cost = computeCost(theta,x,y)
        jvec.append(cost)
        thetahistory.append(list(theta[:,0]))
        for j in range(len(theta)):
            tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(myx=x,mytheta=tmptheta) - y)*np.array(x[:,j]).reshape(m,1))
    theta = tmptheta
    #theta 最后一次迭代 的参数
    #thetahistory历史迭代的参数
    #jvec 迭代后所有的 cost
    return theta, thetahistory, jvec
# re_theta,thetahistory,jevc = descendGradient(x,y,initial_theta)
# theta = re_theta.reshape(3,)
# print(jevc[-1:])
boundary_xs = np.array([np.min(x[:,1]), np.max(x[:,1])])
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)
#w1x1+w2x2+b = 0
#x2 = (-1./w2)*(b+w1x1) => (-1./theta[2])*(theta[0]+theta[1]*x1)
plotData()
plt.plot(boundary_xs,boundary_ys,'b-',label=f'{float(theta[1]):.2f}x1+{float(theta[2]):.2f}x2+{float(theta[0]):.2f}=0')
plt.legend()
plt.show()
# print(h(theta,np.array([1, 45.,85.])))
# print(np.array([1, 45.,85.]))

def makePrediction(mytheta,myx):
    return h(mytheta=mytheta,myx=myx) >=0.5

mp =  makePrediction(theta,pos)

pos_correct = float(np.sum(makePrediction(theta,pos)))
neg_correct = float(np.sum(np.invert(makePrediction(theta,neg))))
tot = len(pos)+len(neg)
print(tot)
prcnt_correct = float(pos_correct+neg_correct)/tot
print("Fraction of training samples correctly predicted: %f." % prcnt_correct )
