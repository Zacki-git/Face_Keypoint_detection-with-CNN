from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import copy
import runpy
import numpy as np
import os
import cv2
import re
from data_myself import get_train_test_set
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Backbone:
        # in_channel, out_channel, kernel_size, stride, padding
        # block 1
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2, 0)
        # block 2
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        # block 3
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        # block 4
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        # points branch
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)
        # commen used
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        self.avg_Pool = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.Dropout = nn.Dropout(0.3)  # dropout p * 100%
        self.BN1_1 = nn.BatchNorm2d(8)  # num features, with learnable parameters
        self.BN2_1 = nn.BatchNorm2d(16)
        self.BN3_1 = nn.BatchNorm2d(24)

    def forward(self, x):
        # block 1
        x = self.avg_Pool(self.prelu1_1(self.conv1_1(x)))
        #x = self.avg_Pool(self.BN1_1(self.prelu1_1(self.conv1_1(x))))  # conv --> relu --> BN --> Pool

        # block 2
        x = self.prelu2_1(self.conv2_1(x))
        #x = self.BN2_1(self.prelu2_1(self.conv2_1(x)))  # conv --> relu --> BN

        x = self.prelu2_2(self.conv2_2(x))
        x = self.avg_Pool(x)

        # block 3
        x = self.prelu3_1(self.conv3_1(x))
        #x = self.BN3_1(self.prelu3_1(self.conv3_1(x)))  # conv --> relu --> BN

        x = self.prelu3_2(self.conv3_2(x))
        x = self.avg_Pool(x)

        # block 4
        x = self.prelu4_1(self.conv4_1(x))

        # points branch
        ip3 = self.prelu4_2(self.conv4_2(x))
        ip3 = ip3.view(-1, 4 * 4 * 80)

        #ip3 = self.preluip1(self.ip1(self.Dropout(ip3)))    # Dropout_1   flatten --> Dropout --> IP1 --> PReLu
        ip3 = self.preluip1(self.ip1(ip3))
        #ip3 = self.Dropout(self.preluip1(self.ip1(ip3)))    # Dropout_2   flatten --> IP1 --> PReLu --> Dropout

        ip3 = self.preluip2(self.ip2(ip3))
        ip3 = self.ip3(ip3)

        return ip3


def train(args, train_loader, valid_loader, model, criterion, optimizer, device, scheduler):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
        point = 0
    if args.checkpoint != "":
        model.load_state_dict(torch.load(args.checkpoint))
        print("Training from checkpoint %s" % args.checkpoint)
        point = int(re.findall(r"\d+", args.checkpoint)[-1]) + 1

    epoch = args.epochs
    criterion = criterion

    log_path = os.path.join(args.save_directory,
                            "log_info" + '_' + str(point) + '_' + str(point + epoch - 1) + '.txt')
    if (os.path.exists(log_path)):
        os.remove(log_path)

    train_losses = []
    valid_losses = []

    for epoch_id in range(epoch):
        # training the model
        model.train()
        log_lines = []
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']

            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            output_pts = model(input_img)

            loss = criterion(output_pts, target_pts)

            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                for param_group in optimizer.param_groups:
                    log_lr = "Current learning rate is: {}".format(param_group['lr'])
                log_line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                    (epoch_id + point),
                    batch_idx * len(img),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
                )
                print(log_line)
                print(log_lr)
                log_lines.append(log_line)
                log_lines.append(log_lr)

        scheduler.step()
        print("scheduler Lr = ", scheduler.get_lr())
        # scheduler.step（）按照Pytorch的定义是用来更新优化器的学习率的，
        # 一般是按照epoch为单位进行更换，即多少个epoch后更换一次学习率，
        # 因而scheduler.step()放在epoch这个大循环下。

        train_losses.append(loss)

        # validate the model
        valid_mean_pts_loss = 0.0
        model.eval()  # prep model for evaluation
        with torch.no_grad():  # 验证模式参数不自动求导
            valid_batch_cnt = 0
            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)

                valid_loss = criterion(output_pts, target_pts)
                valid_mean_pts_loss += valid_loss.item()

            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            log_line = 'Valid: pts_loss: {:.6f}'.format(valid_mean_pts_loss)
            print(log_line)
            log_lines.append(log_line)
            valid_losses.append(valid_mean_pts_loss)
        print('====================================================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id + point) + '.pt')
            torch.save(model.state_dict(), saved_model_name)

        # write log info
        with open(log_path, "a") as f:
            for line in log_lines:
                f.write(line + '\n')
    return train_losses, valid_losses


def test(valid_loader, model, criterion, device):
    model.eval()
    valid_mean_pts_loss = 0.0
    valid_batch_cnt = 0
    result = []
    for batch_idx, batch in enumerate(valid_loader):
        valid_batch_cnt += 1
        valid_img = batch['image']
        input_img = valid_img.to(device)
        # ground truth
        landmarks = batch['landmarks']
        target_pts = landmarks.to(device)
        # result
        output_pts = model(input_img)
        valid_loss = criterion(output_pts, target_pts)
        valid_mean_pts_loss += valid_loss.item()
        device2 = torch.device('cpu')
        output_pts = output_pts.to(device2)
        for i in range(len(valid_img)):
            sample = {
                'image': valid_img[i],
                'landmarks': output_pts[i],
                'landmarks_truth': landmarks[i]
            }
            result.append(sample)
    # 计算平均loss值
    valid_mean_pts_loss /= valid_batch_cnt * 1.0
    return result, valid_mean_pts_loss


def Finetune(args, train_loader, valid_loader, model, criterion, device):
    # load trained model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    if args.checkpoint == "":
        print("Please input the pretrained model in args.checkpoint ")

    if args.checkpoint != "":
        model.load_state_dict(torch.load(args.checkpoint))
        print("Finetuning the models from checkpoint %s" % args.checkpoint)
        point = int(re.findall(r"\d+", args.checkpoint)[-1]) + 1  # re.findall(r"\d+\.?\d*",string)

    epoch = args.epochs
    criterion = criterion

    train_losses = []
    valid_losses = []
    log_path = os.path.join(args.save_directory,
                            "log_info" + '_' + str(point) + '_' + str(point + epoch - 1) + '.txt')
    if (os.path.exists(log_path)):
        os.remove(log_path)

    # training the model
    for para in list(model.parameters())[0:2]:
        print(para)
        # para.requires_grad = False

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # [model.ip2.weight, model.ip2.bias]
    print("only last layer will be trained, other layers are frozen")
    for epoch_id in range(epoch):
        # training the model
        model.train()
        log_lines = []
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']

            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            output_pts = model(input_img)

            # get loss
            loss = criterion(output_pts, target_pts)

            # do BP automatically
            loss.backward()  # 计算出每个神经元节点的导数
            optimizer.step()  # 更新所有参数

            # show log info
            if batch_idx % args.log_interval == 0:
                for param_group in optimizer.param_groups:
                    log_lr = "Current learning rate is: {}".format(param_group['lr'])
                log_line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\t pts_loss: {:.6f}'.format(
                    (epoch_id + point),
                    batch_idx * len(img),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()

                )
                print(log_line)
                print(log_lr)
                log_lines.append(log_line)
                log_lines.append(log_lr)
        train_losses.append(loss)

        # validate the model
        valid_mean_pts_loss = 0.0
        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0
            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)

                valid_loss = criterion(output_pts, target_pts)
                valid_mean_pts_loss += valid_loss.item()

            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            log_line = 'Valid: pts_loss: {:.6f}'.format(valid_mean_pts_loss)
            print(log_line)
            log_lines.append(log_line)
            valid_losses.append(valid_mean_pts_loss)
        print('====================================================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory,
                                            'detector_epoch' + '_' + str(epoch_id + point) + '_' + "Finetune" + '.pt')
            torch.save(model.state_dict(), saved_model_name)  # 保存神经网络里所有的参数
        # write log info
        with open(log_path, "a") as f:
            for line in log_lines:
                f.write(line + '\n')
    return train_losses, valid_losses


def predict(model, valid_loader, device):
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            img = batch['image']
            landmark = batch['landmarks']
            img = img.to(device)
            print('i: ', i)
            # generated
            output_pts = model(img)
            device2 = torch.device('cpu')
            output_pts = output_pts.to(device2)
            outputs = output_pts.numpy()[0]
            print('outputs: ', outputs)
            x = list(map(int, outputs[0: len(outputs): 2]))
            y = list(map(int, outputs[1: len(outputs): 2]))
            landmarks_generated = list(zip(x, y))
            # truth
            landmark = landmark.numpy()[0]
            x = list(map(int, landmark[0: len(landmark): 2]))
            y = list(map(int, landmark[1: len(landmark): 2]))
            landmarks_truth = list(zip(x, y))

            device2 = torch.device('cpu')
            img = img.to(device2)
            img = img.numpy()[0].transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for landmark_truth, landmark_generated in zip(landmarks_truth, landmarks_generated):
                cv2.circle(img, tuple(landmark_truth), 2, (0, 0, 255), -1)
                cv2.circle(img, tuple(landmark_generated), 2, (0, 255, 0), -1)

            cv2.imshow(str(i), img)
            key = cv2.waitKey()
            if key == 27:
                exit()
            cv2.destroyAllWindows()


def loss_show(train_losses, valid_losses, args):
    x = np.arange(0, args.epochs)
    train_losses = np.array(train_losses)
    t_loss = np.c_[x, train_losses]
    valid_losses = np.array(valid_losses)
    v_loss = np.c_[x, valid_losses]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots(nrows=1, ncols=1)
    ax.plot(t_loss[:, 0], t_loss[:, 1], color='red')
    ax.plot(v_loss[:, 0], v_loss[:, 1], color='green')
    plt.show()


def result_show(data):
    #indexes = np.random.randint(0, len(data), 3)
    indexes = [333,113,222]
    fig = plt.figure(figsize=(20, 10))
    axes = fig.subplots(nrows=1, ncols=3)
    for i in range(3):
        sample = data[indexes[i]]
        ax = axes[i]
        img = sample['image']
        img = img[0]
        landmarks = sample['landmarks']
        landmarks = landmarks.reshape(-1, 2)
        gt_lms = sample['landmarks_truth']
        gt_lms = gt_lms.reshape(-1, 2)
        ax.imshow(img, cmap='gray')
        ax.scatter(landmarks[:, 0], landmarks[:, 1], s=10, c='r')
        ax.scatter(gt_lms[:, 0], gt_lms[:, 1], s=10, c='g')
    plt.show()


def main_test():
    parser = argparse.ArgumentParser(description="Detector_myself")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training(default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="N",
                        help="input batch size for testing(default: 64)")
    parser.add_argument("--epochs", type=int, default=250, metavar="N", help="number of epochs to train(default: 100)")
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=117, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models', help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='test',
                        help='training, predicting or finetuning')  # Train/train, Predict/predict, Finetune/finetune
    parser.add_argument('--checkpoint', type=str,
                        default="",  # trained_models\\detector_epoch_399.pt
                        help='run the training from specified checkpoint')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0

    print('==> Loading Datasets')

    train_set, test_set = get_train_test_set()
    # DataLoader 对给定的数据进行批次训练
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)

    print('==> Building Model')
    # For single GPU
    model = Net().to(device)
    if args.phase == 'Test' or args.phase == 'test' or \
            args.phase == 'Predict' or args.phase == 'predict' or \
            args.phase == 'Finetune' or args.phase == 'finetune':
        model_name = os.path.join(args.save_directory,
                                            'detector_epoch'+'_'+str(249)+'.pt')
        model.load_state_dict(torch.load(model_name))
    # loss funktion
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss()

    # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,nesterov = True)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)
    # optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    # optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Scheduler Step
    scheduler= optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5, last_epoch=-1)

    # step_size(int) - 学习率下降间隔数,若为30,则会在30, 60, 90…个step时，将学习率调整为lr * gamma
    # gamma(float) - 学习率调整倍数，默认为0.1倍，即下降10倍
    # last_epoch(int) - 上一个epoch数，这个变量用来指示学习率是否需要调整。
    # 当last_epoch符合设定的间隔时，就会对学习率进行调整,当为 - 1时,学习率设置为初始值

    if args.phase == "Train" or args.phase == "train":
        print('==> Start Training')
        train_losses, valid_losses = train(args, train_loader, valid_loader, model, criterion, optimizer, device, scheduler)
        print("Learning Rate:", args.lr, "Epoch:", args.epochs, "Seed:",
              args.seed, "Batch_Size:", args.batch_size, "Optimizer:", optimizer)
        print('====================================================')

    elif args.phase == 'Test' or args.phase == "test":
        print('==> Testing')
        with torch.no_grad():
            result, valid_mean_pts_loss = test(valid_loader, model, criterion, device)
            print(valid_mean_pts_loss)
        # 利用预测关键点随机作出图像与真实值对比
        result_show(result)

    elif args.phase == 'Finetune' or args.phase == "finetune":
        print('==> Finetune')
        train_losses, valid_losses = Finetune(args, train_loader, valid_loader, model, criterion, device)
        print("Learning Rate:", args.lr, "Epoch:", args.epochs, "Seed:",
              args.seed, "Batch_Size:", args.batch_size)
        print('====================================================')

    elif args.phase == 'Predict' or args.phase == "predict":
        print('==> Predict')
        predict(model, valid_loader, device)


if __name__ == '__main__':
    main_test()
