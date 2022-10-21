

import argparse
from ast import arg
import torch
import torch.nn as nn
import torch.optim as optim
from utils import WarmUpLR
import os


def train(epoch):
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()
        
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'Training Epoch:{epoch}[{batch_index*args.batch_size+len(images)}/{len(training_loader.dataset)}]\t \
                    Loss:{loss.item()}, LR:{optimizer.param_groups[0]["lr"]}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net type')
    parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size of dataloader')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--epoch', type=int, default=300, help='training epoch')
    parser.add_argument('--save_epoch', type=int, default=20, help='save model epoch')
    args = parser.parse_args()

    # 模型
    net = None

    # 数据
    training_loader = None
    test_loader = None

    # 损失函数
    loss_function = nn.CrossEntropyLoss()

    # 优化函数
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # 不同epoch阶段，使用不同学习率
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    iter_per_epoch = len(training_loader)

    # 
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # 保存路径
    checkpoint_path = None
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    model_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    # use tensorboad
    epochs = args.epoch 
    best_acc = 0.0
    for epoch in range(1, epochs):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        
        train(epoch)
        acc = eval_training(epoch)

        if epoch > 120 and best_acc < acc:
            torch.save(net.state_dict(), model_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue
        
        if not epoch % args.save_epoch:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    
        

