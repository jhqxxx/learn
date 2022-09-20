import torch
import matplotlib.pyplot as plt

N_SAMPLES = 20
N_HIDDEN = 300
# torch.linspace: 功能类似 np.range
# torch.unsqueeze:在指定axis增加维度1 | torch.squeeze: 删除指定axis为1的维度   
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
# torch.normal:返回从单独的正态分布中提取的随机数的张量，均值为mean, 标准差为std
#   当mean和std为多维时，达到让每个值服从不同的正态分布
y = x + 0.3*torch.normal(mean=torch.zeros(N_SAMPLES, 1), std=torch.ones(N_SAMPLES, 1))

test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(mean=torch.zeros(N_SAMPLES, 1), std=torch.ones(N_SAMPLES, 1))

# # plt.scatter: 绘制散点图
# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# # plt.legend: 创建图例，并显示label信息，loc设置图例位置
# plt.legend(loc='upper left')
# # plt.ylim/plt.xlim: 设置y轴/x轴数值显示范围
# plt.ylim((-2.5, 2.5))
# plt.show()

def dropout_layer(X, dropout):
    '''该函数以dropout的概率丢弃张量输入X中的元素'''
    assert dropout >= 0 and dropout <= 1
    if dropout == 1:
        return X
    elif dropout == 0:
        return torch.zeros_like(X)
    else:
        mask = (torch.randn(X.shape) > dropout).float()
        return mask * X / (1.0 - dropout)


net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1)
)

net_dropout = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    # dropout一般放在激活函数之后，因为对于部分激活函数输入为0，输出不一定为0，可能起不到效果
    torch.nn.Dropout(0.5),    
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(N_HIDDEN, 1)
)

optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_dropout.parameters(), lr=0.01)

loss_func = torch.nn.MSELoss()

# matplotlib显示模式默认为阻塞模式：在plt.show()之后，程序会暂停在那儿，不会继续执行
# ion()使matplotlib的显示模式转换为交互模式：即使代码中遇到plt.show():代码还是会继续执行
# 如果在脚本中使用ion()开启交互模式，但没有使用ioff()关闭的话，图像会一闪而过，并不会停留
plt.ion()

for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropout(x)
    loss_ofit = loss_func(pred_ofit, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:
        # change to eval mode in order to fix drop out effect
        # 在测试模型时加入net.eval():不启用BatchNormalization和Dropout, 
        # 此时pytorch会自动把BN和Droput固定住
        # 不然的话，有输入数据，即使不训练，它也会改变权值，这是BatchNormalization层的特性
        #       也会有数据被dropout，导致预测结果有问题
        net_overfitting.eval()
        net_dropout.eval()  # parameters for dropout differ from train mode

        # plotting
        # cla(): 清除axes,
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropout(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        # plot(): 绘制点和线
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        # text(): 绘制文字说明
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(), fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((-2.5, 2.5))
        # pause(0.1): 暂停间隔0.1s, 如果有活动图形，它将在暂停之前更新并显示，
        # 并且GUI事件循环(如果有)将在暂停期间运行
        plt.pause(0.1)

        # change back to train mode
        # 训练时net.train()
        net_overfitting.train()
        net_dropout.train()

# ioff()关闭交互模式，使得画面可以停留
plt.ioff()
plt.show()

