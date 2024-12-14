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
    plt.show()

plotData()