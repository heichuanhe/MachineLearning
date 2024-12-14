import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import LinearRegressionModel as LRM
datafile = 'data/ex1data1.txt'
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1),unpack=True)
x = np.transpose(np.array(cols[:-1]))
y = np.transpose(cols[-1:])
m = y.size
x = np.insert(x,0,1,axis=1)

fig,a = plt.subplots(nrows=1,ncols=2)
plt.subplots_adjust(bottom=0.2)  # 为滑块腾出空间
fig.set_size_inches(10,6)
a[0].plot(x[:,1],y[:,0],'rx',markersize=10)
a[0].grid(True) #Always plot.grid true!
a[0].set_ylabel('Profit in $10,000s')
a[0].set_xlabel('Population of City in 10,000s')


iterations =1500   #迭代次数
alpha = 0.01  #学习率/步幅


initial_theta = np.zeros((x.shape[1],1))


theta, thetahistory, jvec = LRM.descentGradient(alpha,x,y,initial_theta,iterations)
# print(jvec)
print(theta)
theta_array = np.array(thetahistory)
# for i in range(len(theta_array)-1):
#     x_f = np.linspace(0,30)
#     y_f = x_f*theta_array[i][1]+theta_array[i][0]
#     plt.plot(x_f,y_f)
# for i in range(len(theta_array)-1):
#     x_f = np.linspace(0,30)
#     y_f = x_f*theta_array[i][1]+theta_array[i][0]
#     a[0].plot(x_f,y_f)
def plotLinear():
    pass




# x_f = [0,-theta[0,0]/theta[1,0]]
# y_f = [theta[0,0],0]
# x_f = np.linspace(0,30)
# y_f = x_f*theta[1,0]+theta[0,0]
# a[0].plot(x_f,y_f)


# jvec_array = np.array(jvec)
# plt.subplot(122)
# plt.plot(theta_array[:,1],jvec_array)
# plt.xlabel("w")
# plt.ylabel("J(w,b)")
# plt.show()
x_f = np.linspace(0,30)
y_f = x_f*theta_array[0,1]+theta_array[0,0]
line, = a[0].plot(x_f,y_f,label=f'y={theta_array[0,1]}*x+{theta_array[0,0]}')
a[0].legend()
jvec_array = np.array(jvec)

curve = a[1].plot(theta_array[:,1],jvec_array)
a[1].set_xlabel("w")
a[1].set_ylabel("J(w,b)")
# 创建用于改变斜率的滑块
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]指定滑块位置和大小
slider_k = Slider(ax_slider, 'Slope (k)', 1, iterations, valinit=1)

#定义更新函数，用于根据滑块的值更新图像
def update(val):
    k = int(slider_k.val)-1
    line.set_ydata(theta_array[k,1] * x_f + theta_array[k,0])
    line.set_label(f'y={theta_array[k,1]:.2f}*x+{theta_array[k,0]:.2f}')
    a[0].legend()
    fig.canvas.draw_idle()

#将滑块的事件与更新函数绑定
slider_k.on_changed(update)



plt.show()