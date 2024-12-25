import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.special import expit
def hypothesis(x,theta):
    return expit(np.dot(x,theta))

def computeCost(x,y,theta):
    m = x.shape[0]
    error = hypothesis(x,theta)-y
    #error 和y的维度是一样的都是m*1
    error_squard = np.sum(error*error)
    cost = float(1./2*m)*error_squard
    return cost


""" 
x:样本特征值
y:样本标签值
alpha:学习率
theta_start:初始参数
iterations:迭代次数"""
def descentGradient(alpha,x,y,theta_start,iterations):
	m = x.shape[0]
	theta = theta_start
	jvec = []  #用于存放每次迭代后计算的损失函数值
	thetahistory = []  #用于存放每次梯度下降后学习得到的参数
	for i in range(iterations):
		cost = computeCost(x,y,theta)
		jvec.append(cost)
		thetahistory.append(list(theta[:,0]))
		for j in range(len(theta)):
			theta[j] = theta[j]-(alpha/m)*np.sum((hypothesis(x,theta)-y)*np.array(x[:,j]).reshape(x.shape[0],1))
			
	return theta,thetahistory,jvec
def model_fit(alpha,x,y,theta_start,iterations):
	m = x.shape[0]
	theta = theta_start
	
	for i in range(iterations):
		cost = computeCost(x,y,theta)
		
		for j in range(len(theta)):
			theta[j] = theta[j]-(alpha/m)*np.sum((hypothesis(x,theta)-y)*np.array(x[:,j]).reshape(x.shape[0],1))		
	return theta
def model_predict(theta,x):
	y_pred = np.dot(x,theta)
	return y_pred

y = np.arange(10)
y_pred = np.arange(20,step=2)

def model_mean_squared_error(y_train,y_pred):
	n = y_train.shape[0]
	mse = (1/n)*np.dot(np.array(y_pred-y_train).T,np.array(y_pred-y_train))
	return mse
