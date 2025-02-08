'''
Descripttion: 
version: 
Author: jhq
Date: 2022-10-22 17:26:00
LastEditors: Please set LastEditors
LastEditTime: 2022-11-24 11:06:10
'''
import os
from os.path import join
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision
import timm
from torch.utils.tensorboard import SummaryWriter
from timm.utils import accuracy
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from mae_util import misc
from mae_util.misc import NativeScalerWithGradNormCount as NativeScaler
import math

from PIL import Image
import argparse
import glob
import random
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        
        output = torch.nn.functional.softmax(output, dim=-1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print(f'Acc@1{metric_logger.acc1.global_avg}, Acc@5{metric_logger.acc5.global_avg}, loss:{metric_logger.loss.global_avg}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, 
                    max_norm=0, log_writer=None, args=None):
    model.train(True)
    print_freq = 2
    if args is not None:
        accum_iter = args.accum_iter
    else: 
        accum_iter = 1
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)
        warmup_lr = args.lr
        optimizer.param_groups[0]['lr'] = warmup_lr
        loss = criterion(outputs, targets)
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss_value = loss.item()
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if not math.isfinite(loss_value):
            print(f'loss:{loss_value}, stopping training')
            sys.exit(1)
        
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', warmup_lr, epoch_1000x)
            print(f'epoch:{epoch}, step:{data_iter_step}, loss:{loss}, lr:{warmup_lr}')


def build_transform(is_train, args):
    if is_train:
        print('train transoform')
        return transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ToTensor(),
            ]
        )
    print("eval transform")
    return transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ]
    )

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    path = join(args.root_path, 'train' if is_train else 'val')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    return dataset

def main(args, mode='train', test_image_path=''):
    print(f'mode:{mode}')
    if mode == 'train':
        # 数据
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
                        dataset_train, sampler=sampler_train,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_mem,
                        drop_last=True)
        data_loader_val = torch.utils.data.DataLoader(
                        dataset_val, sampler=sampler_val,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_mem,
                        drop_last=False)
        
        # 模型
        model = timm.create_model('resnet18', pretrained=True, num_classes=36, drop_rate=0.1, drop_path_rate=0.1)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'number of trainable params (M):{n_parameters/1.e6}')

        # 目标函数
        criterion = torch.nn.CrossEntropyLoss()
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # 日志
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

        model.cuda()

        for epoch in range(args.start_epoch, args.epochs):
            print(f'Epoch:{epoch}')
            if epoch % 1 == 0:
                print('Evaluating...')
                model.eval()
                test_stats = evaluate(data_loader_val, model, device)
                print(f'Accuracy of the network on the{len(dataset_val)} test images:{test_stats["acc1"]:.1f}%')

                if log_writer is not None:
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
                model.train()
            print("Training...")
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch+1,
                loss_scaler, None,
                log_writer=log_writer, args=args
            )
            if args.output_dir:
                print('Saving checkpoints...')
                misc.save_model(args=args, model=model,
                                model_without_ddp=model, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch)
    else:
        model = timm.create_model('resnet18', pretrained=True, num_classes=36, drop_rate=0.1, drop_path_rate=0.1)
        class_dict = {'apple': 0, 'banana': 1, 'beetroot': 2, 'bell pepper': 3, 'cabbage': 4, 
                'capsicum': 5, 'carrot': 6, 'cauliflower': 7, 'chilli pepper': 8, 'corn': 9, 
                'cucumber': 10, 'eggplant': 11, 'garlic': 12, 'ginger': 13, 'grapes': 14, 
                'jalepeno': 15, 'kiwi': 16, 'lemon': 17, 'lettuce': 18, 'mango': 19, 'onion': 20, 
                'orange': 21, 'paprika': 22, 'pear': 23, 'peas': 24, 'pineapple': 25, 'pomegranate': 26, 
                'potato': 27, 'raddish': 28, 'soy beans': 29, 'spinach': 30, 'sweetcorn': 31, 
                'sweetpotato': 32, 'tomato': 33, 'turnip': 34, 'watermelon': 35}
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok=True)
        loss_scaler = NativeScaler()
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
        model.eval()
        image = Image.open(test_image_path).convert('RGB')
        img = img.resize((args.input_size, args.input_size), Image.ANTIALIAS)
        image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
        output = torch.nn.functional.softmax(output, dim=-1)
        class_idx = torch.argmax(output, dim=1)[0]
        score = torch.max(output, dim=1)[0][0]

        print(f'image path is {test_image_path}')
        print(f'score:{score.item()}, class id is:{class_idx.item()},class name:{list(class_dict.keys())[list(class_dict.values()).index(class_idx)]}')

def get_args_parser():
    parser = argparse.ArgumentParser('training args', add_help=False)
    parser.add_argument('--batch_size', default=36, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--root_path', default=r'C:\jhq\dataset\garden_staff')
    parser.add_argument('--output_dir', default=r'C:\jhq\save\model')
    parser.add_argument('--log_dir', default=r'C:\jhq\save\log')
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    mode = 'train'
    if mode == 'train':
        main(args, mode=mode)
    else:
        images = glob.glob('C:\jhq\dataset\garden_staff\test\*\*.jpg')
        random.shuffle(images)

        for image in images:
            main(args, mode=mode, test_image_path=image)