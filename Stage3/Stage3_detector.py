from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from data_myself_stage3 import get_train_test_set
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积参数：in_channel, out_channel, kernel_size, stride, padding
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
        # landmarks branch
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)
        # cls branch
        self.conv4_2_cls = nn.Conv2d(40, 40, 3, 1, 1)
        self.ip1_cls = nn.Linear(4 * 4 * 40, 128)
        self.ip2_cls = nn.Linear(128, 128)
        self.ip3_cls = nn.Linear(128, 2)

        # common used
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.bn1_1 = nn.BatchNorm2d(8)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.bn2_2 = nn.BatchNorm2d(16)
        self.bn3_1 = nn.BatchNorm2d(24)
        self.bn3_2 = nn.BatchNorm2d(24)
        self.bn4_1 = nn.BatchNorm2d(40)
        self.dropout = nn.Dropout(0.5)
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)
        # landmarks branch
        self.prelu4_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        # cls branch
        self.prelu4_2_cls = nn.PReLU()
        self.preluip1_cls = nn.PReLU()
        self.preluip2_cls = nn.PReLU()


    def forward(self, x):  # x is input
        # block 1
        x = self.ave_pool(self.bn1_1(self.prelu1_1(self.conv1_1(x))))
        # block 2
        x = self.ave_pool(self.bn2_2(self.prelu2_2(self.conv2_2(self.bn2_1(self.prelu2_1(self.conv2_1(x)))))))
        # block 3
        x = self.ave_pool(self.bn3_2(self.prelu3_2(self.conv3_2(self.bn3_1(self.prelu3_1(self.conv3_1(x)))))))
        # block 4
        x = self.bn4_1(self.prelu4_1(self.conv4_1(x)))

        # landmarks branch
        x_lb = self.prelu4_2(self.conv4_2(x))
        x_lb = self.dropout(x_lb.view(-1, 4 * 4 * 80))
        ip1_lb = self.preluip1(self.ip1(x_lb))
        ip2_lb = self.preluip2(self.ip2(ip1_lb))
        ip3_lb = self.ip3(ip2_lb)

        # cls branch
        x_cls = self.prelu4_2_cls(self.conv4_2_cls(x))
        x_cls = self.dropout(x_cls.view(-1, 4 * 4 * 40))
        ip1_cls = self.preluip1_cls(self.ip1_cls(x_cls))
        ip2_cls = self.preluip2_cls(self.ip2_cls(ip1_cls))
        ip3_cls = self.ip3_cls(ip2_cls)
        #ip3_cls = F.softmax(ip3_cls, dim = 1)
        return ip3_cls, ip3_lb


def main_test():
    # 1、进行参数设置
    parser = argparse.ArgumentParser(description="Detector_myself")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training(default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=64, metavar="N",
                        help="input batch size for testing(default: 64)")
    parser.add_argument("--epochs", type=int, default=300, metavar="N",
                        help="number of epochs to train(default: 100)")
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=117, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=80, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',
                        # Train/train, Test/test
                        help='training, test')
    parser.add_argument('--checkpoint', type=str,
                        default='trained_models\\detector_epoch_13.pt',
                        help='run the specified checkpoint model')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='start training at checkpoint')
    args = parser.parse_args()

    # 2、基本设置
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0
    # 3、读取数据
    print('==> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    # 4、将数据/网络传入CPU/GPU
    print('==> Building Model')
    model = Net().to(device)
    if args.phase == 'Test' or args.phase == 'test':
        model_name = args.checkpoint
        model.load_state_dict(torch.load(model_name))
        model.eval()

    # 5、定义损失函数和优化器

    # parameter of weighted cross entropy
    weights = [5077.0/7250.0,2173.0/7250.0]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion1 = nn.CrossEntropyLoss(weight=class_weights) # binary classification  num(0), num (1)
    criterion2 = nn.SmoothL1Loss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # 6、Scheduler Step
    # scheduler= optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
    # step_size(int) - 学习率下降间隔数,若为30,则会在30, 60, 90…个step时，将学习率调整为lr * gamma
    # gamma(float) - 学习率调整倍数，默认为0.1倍，即下降10倍
    # last_epoch(int) - 上一个epoch数，这个变量用来指示学习率是否需要调整。
    # 当last_epoch符合设定的间隔时，就会对学习率进行调整,当为 - 1时,学习率设置为初始值

    # 7、定义程序所处阶段
    if args.phase == "Train" or args.phase == "train":

        print('==> Start Training')
        if args.retrain:
            model.load_state_dict(torch.load(args.checkpoint))
            print("Training from checkpoint %s" % args.checkpoint)
        train_losses, valid_losses = train(args, train_loader, valid_loader,
                                           model, criterion1, criterion2, optimizer, device)
        print("Learning Rate:", args.lr, "Epoch:", args.epochs, "Seed:",
              args.seed, "Batch_Size:", args.batch_size, "Optimizer:", optimizer)
        print('====================================================')
        loss_show(train_losses, valid_losses, args)

    elif args.phase == 'Test' or args.phase == "test":
        print('==> Testing')
        with torch.no_grad():
            result, valid_mean_pts_loss, accuracy= test(valid_loader, model, criterion1, criterion2, device)
            print(valid_mean_pts_loss)
            print(accuracy)
        # 利用预测关键点随机作出图像与真实值对比
        result_show(result)

def train(args, train_loader, valid_loader, model, criterion1, criterion2, optimizer, device):
    # 设定保存
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    # 设定训练次数、损失函数
    epoch = args.epochs
    # monitor training loss
    train_losses = []
    valid_losses = []
    log_path = os.path.join(args.save_directory, 'log_info.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    # 开始训练模型
    for epoch_id in range(epoch):
        # training the model
        model.train()
        log_lines = []



        #params for statistic
        train_pred_correct = 0
        train_pred_correct_zero = 0
        train_pred_correct_one = 0
        train_zero_num = 0
        train_one_num = 0
        valid_pred_correct = 0
        valid_pred_correct_zero = 0
        valid_pred_correct_one = 0
        valid_zero_num = 0
        valid_one_num = 0

        for batch_idx, batch in enumerate(train_loader):
            # input
            img = batch['image']
            input_img = img.to(device)
            # ground truth
            landmarks = batch['landmarks']
            cls = batch['class']
            target_cls = cls.to(device)
            target_pts = landmarks.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            output_cls, output_pts = model(input_img)

            #statistic
            _, pred_train = torch.max(output_cls.data, 1) #比较dim为1情况下的最大值，返回最大值和最大值对应下标
            # 正样本为[0,1]，负样本为[1,0]
            train_pred_correct += (pred_train == target_cls).sum()
            train_pred_correct_zero += ((pred_train==0)&(target_cls==0)).sum()
            train_pred_correct_one += ((pred_train==1)&(target_cls==1)).sum()
            train_zero_num += (target_cls==0).sum()
            train_one_num += (target_cls==1).sum()

            # First: classification
            loss_cls = criterion1(output_cls, target_cls)

            mask = target_cls.reshape((-1,1)).float()

            # Second: regression
            output_pts = mask * output_pts
            target_pts = mask * target_pts

            loss_pts = criterion2(output_pts,target_pts)
            # Loss
            loss = 1 * (2*loss_cls + 1*loss_pts)
            # do BP automatically
            loss.backward()
            optimizer.step()

            # show log info
            if batch_idx % args.log_interval == 0:
                log_line = 'Train Epoch:{}[{}/{}({:.0f}%)]\t loss:{:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img),  # 批次序号*一批次样本数=已测样本数
                    len(train_loader.dataset),  # 总train_set样本数量
                    100. * batch_idx / len(train_loader),  # 以上两者之比
                    loss.item()
                )
                print(log_line)

                log_lines.append(log_line)
        train_losses.append(loss)

        Train_CLS_accuracy = train_pred_correct.item() / len(train_loader.dataset)
        Train_CLS_zero_accuracy = train_pred_correct_zero.item() / train_zero_num.item()
        Train_CLS_one_accuracy = train_pred_correct_one.item() / train_one_num.item()
        log_line_train_accuracy = 'Train_CLS_accuracy:{:.4f}% ({} / {})\n' \
                            'Train_CLS_one_accuracy:{:.4f}% ({} / {})\n' \
                            'Train_CLS_zero_accuracy:{:.4f}% ({} / {})'.format(
            100 * Train_CLS_accuracy,train_pred_correct.item(),len(train_loader.dataset),
            100 * Train_CLS_one_accuracy,train_pred_correct_one.item(),train_one_num.item(),
            100 * Train_CLS_zero_accuracy,train_pred_correct_zero.item(),train_zero_num.item()
        )

        # 验证（使用测试数据集）
        valid_mean_pts_loss = 0.0

        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0
            for batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                valid_img = valid_img.to(device)
                # ground truth
                valid_landmarks = batch['landmarks']
                valid_cls = batch['class']
                valid_target_cls = valid_cls.to(device)
                valid_target_pts = valid_landmarks.to(device)
                #print("valid_target_cls = ", valid_target_cls, valid_target_cls.shape)

                # result
                valid_output_cls, valid_output_pts = model(valid_img)

                #statistic
                _, pred_valid = torch.max(valid_output_cls.data, 1)

                valid_pred_correct += (pred_valid == valid_target_cls).sum()
                valid_pred_correct_zero += ((pred_valid == 0) & (valid_target_cls == 0)).sum()
                valid_pred_correct_one += ((pred_valid == 1) & (valid_target_cls == 1)).sum()
                valid_zero_num += (valid_target_cls == 0).sum()
                valid_one_num += (valid_target_cls == 1).sum()

                # Valid CLS Loss
                valid_loss_cls = criterion1(valid_output_cls, valid_target_cls)

                valid_mask = valid_target_cls.reshape((-1,1)).float()
                valid_output_pts = valid_mask * valid_output_pts
                valid_target_pts = valid_mask * valid_target_pts
                valid_loss_pts = criterion2(valid_output_pts, valid_target_pts)

                # Loss
                valid_loss = 1 * (10* valid_loss_cls + 1 * valid_loss_pts)
                valid_mean_pts_loss += valid_loss.item()

            # 结论输出
            valid_mean_pts_loss /= valid_batch_cnt * 1.0

            log_line = 'Valid: loss: {:.6f}'.format(valid_mean_pts_loss)
            print(log_line)
            log_lines.append(log_line)
            valid_losses.append(valid_mean_pts_loss)

            Valid_CLS_accuracy = valid_pred_correct.item() / len(valid_loader.dataset)
            Valid_CLS_zero_accuracy = valid_pred_correct_zero.item() / valid_zero_num.item()
            Valid_CLS_one_accuracy = valid_pred_correct_one.item() / valid_one_num.item()
            log_line_valid_accuracy = 'Valid_CLS_accuracy:{:.4f}% ({} / {})\n' \
                                'Valid_CLS_one_accuracy:{:.4f}% ({} / {})\n' \
                                'Valid_CLS_zero_accuracy:{:.4f}% ({} / {})'.format(
                100 * Valid_CLS_accuracy,valid_pred_correct.item(),len(valid_loader.dataset),
                100 * Valid_CLS_one_accuracy,valid_pred_correct_one.item(),valid_one_num.item(),
                100 * Valid_CLS_zero_accuracy,valid_pred_correct_zero.item(),valid_zero_num.item()
            )

            log_lines.append(log_line_train_accuracy)
            print(log_line_train_accuracy)
            log_lines.append(log_line_valid_accuracy)
            print(log_line_valid_accuracy)

        print('=============================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory,
                                            'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
        # write log info
        with open(log_path, "a") as f:
            for line in log_lines:
                f.write(line + '\n')
    return train_losses, valid_losses

# loss作图
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

# 测试，计算测试集上预测关键点
def test(valid_loader, model, criterion1, criterion2, device):
    valid_mean_pts_loss = 0.0
    valid_batch_cnt = 0
    result = []

    valid_pred_correct = 0
    valid_pred_correct_zero = 0
    valid_pred_correct_one = 0
    valid_zero_num = 0
    valid_one_num = 0

    for batch_idx, batch in enumerate(valid_loader):
        valid_batch_cnt += 1
        valid_img = batch['image']
        input_img = valid_img.to(device)
        # ground truth
        landmarks = batch['landmarks']
        cls = batch['class']
        target_cls = cls.to(device)
        target_pts = landmarks.to(device)
        # result
        output_cls, output_pts = model(input_img)
        # loss_cls
        loss_cls = criterion1(output_cls, target_cls)
        # accuracy
        _, pred_valid = torch.max(output_cls.data, 1)
        valid_pred_correct += (pred_valid == target_cls).sum()
        valid_pred_correct_zero += ((pred_valid == 0) & (target_cls == 0)).sum()
        valid_pred_correct_one += ((pred_valid == 1) & (target_cls == 1)).sum()
        valid_zero_num += (target_cls == 0).sum()
        valid_one_num += (target_cls == 1).sum()
        # loss_pts
        mask = target_cls.reshape((-1,1)).float()
        output_pts = mask * output_pts
        target_pts = mask * target_pts
        loss_pts = criterion2(output_pts, target_pts)

        # Loss
        valid_loss = 1 * (10 * loss_cls + 1 * loss_pts)
        valid_mean_pts_loss += valid_loss.item()
        device2 = torch.device('cpu')
        output_pts = output_pts.to(device2)
        output_cls = output_cls.to(device2)
        for i in range(len(valid_img)):
            sample = {
                'image': valid_img[i],
                'class': output_cls[i],
                'landmarks': output_pts[i],
                'landmarks_truth': landmarks[i]
            }
            result.append(sample)
    # 计算loss值
    valid_mean_pts_loss /= valid_batch_cnt * 1.0
    # accuracy
    Valid_CLS_accuracy = valid_pred_correct.item() / len(valid_loader.dataset)
    Valid_CLS_zero_accuracy = valid_pred_correct_zero.item() / valid_zero_num.item()
    Valid_CLS_one_accuracy = valid_pred_correct_one.item() / valid_one_num.item()
    line_accuracy = 'Valid_CLS_accuracy:{:.4f}% ({} / {})\n' \
                    'Valid_CLS_one_accuracy:{:.4f}% ({} / {})\n' \
                    'Valid_CLS_zero_accuracy:{:.4f}% ({} / {})'.format(
        100 * Valid_CLS_accuracy,valid_pred_correct.item() , len(valid_loader.dataset),
        100 * Valid_CLS_one_accuracy,valid_pred_correct_one.item() , valid_one_num.item(),
        100 * Valid_CLS_zero_accuracy,valid_pred_correct_zero.item() ,valid_zero_num.item()
    )

    return result, valid_mean_pts_loss, line_accuracy

# 利用预测关键点作出图像与真实值对比
def result_show(data):
    indexes = np.random.randint(0, len(data), 3)
    fig = plt.figure(figsize=(10, 10))
    axes = fig.subplots(nrows=1, ncols=3)
    for i in range(3):
        sample = data[indexes[i]]
        ax = axes[i]
        img = sample['image']
        img = img[0]
        cls = sample['class']
        if cls[0] < cls[1] :
            landmarks = sample['landmarks']
            landmarks = landmarks.reshape(-1, 2)
            gt_lms = sample['landmarks_truth']
            gt_lms = gt_lms.reshape(-1, 2)
            ax.imshow(img, cmap='gray')
            ax.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='r')
            ax.scatter(gt_lms[:, 0], gt_lms[:, 1], s=5, c='g')
        else:
            ax.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main_test()

