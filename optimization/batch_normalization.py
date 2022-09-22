'''
Descripttion: 
version: 
Author: jhq
Date: 2022-09-20 22:22:06
LastEditors: jhq
LastEditTime: 2022-09-21 23:24:07
'''
from matplotlib.pyplot import axis
import numpy as np

class BatchNormalization:
    '''
    思路：强制性调整输入数据的分布，均值为0，方差为1，
        适当的输入数据，带来适当的激活值分布，保证训练正常进行
    作用：防止过拟合 | 加速收敛
    位置：Affine(仿射层，矩阵乘法，指全连接层)、convolution层后面
    原理：对数据的均值方差进行调整，加速训练收敛
    '''

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None   # Conv层为4维， 全连接层为2维

        # 测试是使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
    
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        
        out = self.__forward(x, train_flg)
        return out.reshape(self.input_shape)
    
    def __forward(self, x, train_fg):
        if self.running_mean == None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        
        if train_fg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))
        
        # gamma: 原始输入x的方差
        # beta: 原始输入x的期望
        out = self.gamma * xn + self.beta
        return out
    
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)
        
        dx = self.__backward(dout)
        dx = dx.reshape(*self.input_shape)
        return dx
    
    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx
