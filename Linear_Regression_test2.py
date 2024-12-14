import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class LinearRegression:
    def __init__(self, master):
        self.master = master
        self.iteration = 10
        self.create_widgets()

    def create_widgets(self):
        # 创建Entry组件
        self.entry = tk.Entry(self.master)
        self.entry.pack()

        # 创建Button组件，点击时调用update_iteration方法
        button = tk.Button(self.master, text="获取Entry值并更新iteration",
                           command=self.update_iteration)
        button.pack()

    def update_iteration(self):
        try:
            # 获取Entry组件中的值并转换为整数
            value = int(self.entry.get())
            self.iteration = value
            print(f"iteration变量已更新为: {self.iteration}")
        except ValueError:
            print("请在Entry中输入有效的整数内容！")

    
root = tk.Tk()

root.title('Linear Regression')
root.geometry('500x500')
frame = tk.Frame(root)
frame.pack()
lr = LinearRegression(frame)

l = tk.Label(frame,text='请输入步数：')
l.pack()
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

    #迭代次数
alpha = 0.01  #学习率/步幅
#h(x) = theta[0] + theta[1]*x
def h(theta,x):
    # theta = [[theta[0]],  
    #           [theta[1]]]
    return np.dot(x,theta)

def computeCost(mytheta,x,y):
    #squared_errors = (predictions-y)**2  + np.sum(squared_errors)
    dot_result = np.dot((h(mytheta,x)-y).T,(h(mytheta,x)-y))
    if dot_result.size == 1:
        single_element = dot_result[0,0]
        return float((1./2*m)) * single_element
    else:
        raise ValueError("Unexpected result shape from np.dot operation")

initial_theta = np.zeros((x.shape[1],1))
result = (h(initial_theta,x) - y)*np.array(x[:,0]).reshape(m,1)
print(len(initial_theta))
def descendGradient(x,y,theta_start):
    theta = theta_start
    jvec = [] #Used to plot cost as function of iteration
    thetahistory = [] #Used to visualize the minimization path later on
    for _ in range(lr.iteration):
        tmptheta = theta
        cost = computeCost(theta,x,y)
        jvec.append(cost)
        thetahistory.append(list(theta[:,0]))
        for j in range(len(theta)):
            tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(theta,x) - y)*np.array(x[:,j]).reshape(m,1))
        theta = tmptheta
    #theta 最后一次迭代 的参数
    #thetahistory历史迭代的参数
    #jvec 迭代后所有的 cost
    return theta, thetahistory, jvec

theta, thetahistory, jvec = descendGradient(x,y,initial_theta)
theta_array = np.array(thetahistory)
x_f = np.linspace(0,30)
y_f = x_f*theta_array[0,1]+theta_array[0,0]
line, = a[0].plot(x_f,y_f,label=f'y={theta_array[0,1]}*x+{theta_array[0,0]}')
a[0].legend()
jvec_array = np.array(jvec)

curve = a[1].plot(theta_array[:,1],jvec_array)
a[1].set_xlabel("w")
a[1].set_ylabel("J(w,b)")


#定义更新函数，用于根据滑块的值更新图像
def update(val):
    k = int(val)-1
    line.set_ydata(theta_array[k,1] * x_f + theta_array[k,0])
    line.set_label(f'y={theta_array[k,1]:.2f}*x+{theta_array[k,0]:.2f}')
    a[0].legend()
    fig.canvas.draw_idle()
s1 = tk.Scale(frame,from_=1,to=lr.iteration,orient='horizontal',command=update)
s1.pack()
canvas = FigureCanvasTkAgg(fig,master=frame)
canvas.draw()
canvas.get_tk_widget().pack()
def on_close():
    # 销毁FigureCanvasTkAgg对象
    canvas.get_tk_widget().destroy()
    # 清除matplotlib的Figure对象
    plt.close(fig)
    # 退出tkinter主循环
    root.quit()
    # 结束程序
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()