import numpy as np

def hypothesis(x,theta):
    return np.dot(x,theta)

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