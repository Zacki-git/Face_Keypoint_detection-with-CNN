from __future__ import print_function, division
import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from Project2_data_myself import get_train_test_set
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 特征提取层采用resnet18 pretrained之后的模型
        # points branch
        self.ip2 = nn.Linear(1000, 128)
        self.ip3 = nn.Linear(128, 42)
        # commen used
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        self.Dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 加入了dropout
        ip1 = self.preluip1(self.Dropout(x))
        ip2 = self.preluip2(self.ip2(ip1))
        ip3 = self.ip3(ip2)

        return ip3


def main_test():
    # 1、进行参数设置
    parser = argparse.ArgumentParser(description="Detector_myself")
    parser.add_argument("--batch-size", type=int, default=16, metavar="N",
                        help="input batch size for training(default: 16)")
    parser.add_argument("--test-batch-size", type=int, default=16, metavar="N",
                        help="input batch size for testing(default: 16)")
    parser.add_argument("--epochs", type=int, default=300, metavar="N",
                        help="number of epochs to train(default: 100)")
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
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
                        # Train/train, Test/test, Predict/predict, Finetune/finetune
                        help='training, test, predicting or finetuning')
    parser.add_argument('--checkpoint', type=str,
                        default='trained_models\\detector_epoch_298.pt',
                        help='run the specified checkpoint model')
    parser.add_argument('--retrain', action='store_true', default=True,
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
    if args.phase == 'Test' or args.phase == 'test' or \
            args.phase == 'Predict' or args.phase == 'predict':
        model_name = args.checkpoint
        model.load_state_dict(torch.load(model_name))
        model.eval()

    # 5、定义损失函数和优化器
    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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
        train_losses, valid_losses = train(args, train_loader, valid_loader, model, criterion, optimizer, device)
        print("Learning Rate:", args.lr, "Epoch:", args.epochs, "Seed:",
              args.seed, "Batch_Size:", args.batch_size, "Optimizer:", optimizer)
        print('====================================================')
        loss_show(train_losses, valid_losses, args)

    elif args.phase == 'Test' or args.phase == "test":
        print('==> Testing')
        with torch.no_grad():
            result, valid_mean_pts_loss = test(valid_loader, model, criterion, device)
            print(valid_mean_pts_loss)
        # 利用预测关键点随机作出图像与真实值对比
        result_show(result)

    elif args.phase == 'Predict' or args.phase == "predict":
        print('==> Predict')
        predict(model, valid_loader, device)


def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    global loss
    # 设定保存
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    # 设定训练次数、损失函数
    epoch = args.epochs
    criterion = criterion
    # monitor training loss
    train_losses = []
    valid_losses = []
    log_path = os.path.join(args.save_directory, 'log_info.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    # 获取已预训练好的resnet模型及参数
    model_pt = models.resnet18(pretrained=True)
    model_pt = model_pt.to(device)
    for param in model_pt.parameters():
        param.requires_grad = False
    # 开始训练自己的模型
    for epoch_id in range(epoch):
        # training the model
        model.train()
        log_lines = []
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            input_img = img.to(device)
            input_image = model_pt(input_img)
            # ground truth
            landmarks = batch['landmarks']
            target_pts = landmarks.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # get output
            output_pts = model(input_image)
            # get loss
            loss = criterion(output_pts, target_pts)
            # do BP automatically
            loss.backward()
            optimizer.step()
            # show log info
            if batch_idx % args.log_interval == 0:
                log_line = 'Train Epoch:{}[{}/{}({:.0f}%)]\t pts_loss:{:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img),  # 批次序号*一批次样本数=已测样本数
                    len(train_loader.dataset),  # 总train_set样本数量
                    100. * batch_idx / len(train_loader),  # 以上两者之比
                    loss.item()
                )
                print(log_line)
                log_lines.append(log_line)
        train_losses.append(loss)
        # 验证（使用测试数据集）
        valid_mean_pts_loss = 0.0
        model.eval()  # prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0
            for batch_idx, batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                input_img = valid_img.to(device)
                input_image = model_pt(input_img)
                # ground truth
                landmarks = batch['landmarks']
                target_pts = landmarks.to(device)
                # result
                output_pts = model(input_image)
                valid_loss = criterion(output_pts, target_pts)
                valid_mean_pts_loss += valid_loss.item()
            # 结论输出
            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            log_line = 'Valid: pts_loss: {:.6f}'.format(valid_mean_pts_loss)
            print(log_line)
            log_lines.append(log_line)
            valid_losses.append(valid_mean_pts_loss)
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
def test(valid_loader, model, criterion, device):
    # 获取已预训练好的resnet模型及参数
    model_pt = models.resnet18(pretrained=True)
    model_pt = model_pt.to(device)
    valid_mean_pts_loss = 0.0
    valid_batch_cnt = 0
    result = []
    for batch_idx, batch in enumerate(valid_loader):
        valid_batch_cnt += 1
        valid_img = batch['image']
        input_img = valid_img.to(device)
        input_image = model_pt(input_img)
        # ground truth
        landmarks = batch['landmarks']
        target_pts = landmarks.to(device)
        # result
        output_pts = model(input_image)
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
    # 计算loss值
    valid_mean_pts_loss /= valid_batch_cnt * 1.0
    return result, valid_mean_pts_loss


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
        landmarks = sample['landmarks']
        landmarks = landmarks.reshape(-1, 2)
        gt_lms = sample['landmarks_truth']
        gt_lms = gt_lms.reshape(-1, 2)
        ax.imshow(img, cmap='gray')
        ax.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='r')
        ax.scatter(gt_lms[:, 0], gt_lms[:, 1], s=5, c='g')
    plt.show()


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


if __name__ == '__main__':
    main_test()

