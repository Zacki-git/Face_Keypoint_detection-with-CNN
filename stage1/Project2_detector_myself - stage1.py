import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import os
from Project2_data_myself import get_train_test_set
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. 定义卷积神经网络
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
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        # points branch
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)  # landmarks

        # common used
        self.prelu1_1 = nn.PReLU()
        self.prelu2_1 = nn.PReLU()
        self.prelu2_2 = nn.PReLU()
        self.prelu3_1 = nn.PReLU()
        self.prelu3_2 = nn.PReLU()
        self.prelu4_1 = nn.PReLU()
        self.prelu4_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.preluip2 = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)  # pool1,pool2,pool3

    def forward(self, x):  # x is input
        # block 1
        x = self.conv1_1(x)
        x = self.prelu1_1(x)
        x = self.ave_pool(x)
        # block 2
        x = self.conv2_1(x)
        x = self.prelu2_1(x)
        x = self.conv2_2(x)
        x = self.prelu2_2(x)
        x = self.ave_pool(x)
        # block 3
        x = self.conv3_1(x)
        x = self.prelu3_1(x)
        x = self.conv3_2(x)
        x = self.prelu3_2(x)
        x = self.ave_pool(x)
        # block 4
        x = self.conv4_1(x)
        x = self.prelu4_1(x)
        x = self.conv4_2(x)
        x = self.prelu4_2(x)
        # points branch
        ip1 = x.view(-1, 4 * 4 * 80)  # flatten
        ip1 = self.preluip1(self.ip1(ip1))
        ip2 = self.preluip2(self.ip2(ip1))
        ip3 = self.ip3(ip2)

        return ip3

# 2. 搭建训练框架
def main_test():
    # 1、进行参数设置
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default=64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default=64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train(default=100)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default=0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum(default=0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed(default=1)')
    parser.add_argument('--log-interval', type=int, default=80, metavar='N',
                        help='how many bathes to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',
                        # Train/train, Test/test, Predict/predict, Finetune/finetune
                        help='training, test, predicting or finetuning')
    parser.add_argument('--epoch-id', type=int, default=399, metavar='N',
                        help='id of the testing model')
    parser.add_argument('--checkpoint', type=str,
                        default='trained_models\\detector_epoch_299.pt',
                        help='run the training from specified checkpoint')
    parser.add_argument('--retrain', action='store_true', default=True,
                        help='start training at checkpoint')

    args = parser.parse_args()

    # 2、程序控制代码
    # 2.1 用同样的随机初始化种子保证结果可复现
    torch.manual_seed(args.seed)
    # 2.2 设置GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # 2.3 读取数据
    print('==> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    # 2.4 将数据/网络传入CPU/GPU
    print('===> Building Model')
    net = Net()
    model = net.to(device)
    if args.phase == 'Test' or args.phase == 'test' or \
            args.phase == 'Predict' or args.phase == 'predict' or \
            args.phase == 'Finetune' or args.phase == 'finetune':
        model_name = os.path.join(args.save_directory,
                                        'detector_epoch' + '_' + str(args.epoch_id) + '.pt')
        model.load_state_dict(torch.load(model_name))

    # 3、定义损失函数和优化器
    # 3.1 均方损失函数
    criterion = nn.MSELoss()
    # 3.2 SGD+Momentum
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.99))

    # 4、定义程序所处阶段
    # 4.1 train
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        if args.retrain:
            model.load_state_dict(torch.load(args.checkpoint))
            print("Training from checkpoint %s" % args.checkpoint)
        train_losses,valid_losses = train(args, train_loader, valid_loader, model, criterion, optimizer, device)
        print('====================================================')
        loss_show(train_losses, valid_losses,args)

    # 4.2 test
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Start Testing')
        with torch.no_grad():
            result, valid_mean_pts_loss = test(valid_loader, model, criterion, device)
            print(valid_mean_pts_loss)
        # 利用预测关键点随机作出图像与真实值对比
        result_show(result)

    # 4.3 finetune
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        train_losses, valid_losses = Finetune(args, train_loader, valid_loader, model, criterion, device)
        print("Learning Rate:", args.lr, "Epoch:", args.epochs, "Seed:",
              args.seed, "Batch_Size:", args.batch_size)
        loss_show(train_losses, valid_losses, args)
        print('====================================================')

    # 4.4 predict
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        predict(model, valid_loader,device)

# 3. 训练阶段
# 3.1 训练函数
def train(args,train_loader,valid_loader,model,criterion,optimizer,device):
    # 设定保存
    global loss
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
    if (os.path.exists(log_path)):
        os.remove(log_path)
    # 训练
    for epoch_id in range(epoch):
        # training the model
        model.train()
        log_lines = []
        if epoch_id > 0:
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        if epoch_id > 300:
            optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.5)

        for batch_idx,batch in enumerate(train_loader):
            img = batch['image']
            input_img = img.to(device)
            # ground truth
            landmarks = batch['landmarks']
            target_pts = landmarks.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # get output
            output_pts = model(input_img)
            # get loss
            loss = criterion(output_pts,target_pts)
            # do BP automatically
            loss.backward()
            optimizer.step()
            # show log info
            if batch_idx % args.log_interval == 0:
                log_line = 'Train Epoch:{}[{}/{}({:.0f}%)]\t pts_loss:{:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img), # 批次序号*一批次样本数=已测样本数
                    len(train_loader.dataset), # 总train_set样本数量
                    100.*batch_idx/len(train_loader), # 以上两者之比
                    loss.item()
                )
                print(log_line)
                log_lines.append(log_line)
        train_losses.append(loss)
        # 验证（使用测试数据集）
        valid_mean_pts_loss = 0.0
        model.eval()# prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0
            for batch_idx,batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                input_img = valid_img.to(device)
                # ground truth
                landmarks = batch['landmarks']
                target_pts = landmarks.to(device)
                # result
                output_pts = model(input_img)
                valid_loss = criterion(output_pts,target_pts)
                valid_mean_pts_loss += valid_loss.item()
            #结论输出
            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            log_line = 'Valid: pts_loss: {:.6f}'.format(valid_mean_pts_loss)
            print(log_line)
            log_lines.append(log_line)
            valid_losses.append(valid_mean_pts_loss)
        print('=============================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory,
                                            'detector_epoch'+'_'+str(epoch_id)+'.pt')
            torch.save(model.state_dict(), saved_model_name)
        # write log info
        with open(log_path, "a") as f:
            for line in log_lines:
                f.write(line + '\n')
    return train_losses,valid_losses

# 3.2 loss作图
def loss_show(train_losses,valid_losses,args):
    x = np.arange(0,args.epochs)
    train_losses = np.array(train_losses)
    t_loss = np.c_[x,train_losses]
    valid_losses = np.array(valid_losses)
    v_loss = np.c_[x, valid_losses]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots(nrows=1, ncols=1)
    ax.plot(t_loss[:,0],t_loss[:,1],color='red')
    ax.plot(v_loss[:, 0], v_loss[:, 1], color='green')
    plt.show()

# 4. 测试阶段
# 4.1 计算测试集上预测关键点
def test(valid_loader,model,criterion,device):
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
    # 计算loss值
    valid_mean_pts_loss /= valid_batch_cnt * 1.0
    return result, valid_mean_pts_loss

# 4.2 利用预测关键点作出图像与真实值对比
def result_show(data):
    indexes = np.random.randint(0,len(data),3)
    fig = plt.figure(figsize=(10,10))
    axes = fig.subplots(nrows=1,ncols=3)
    for i in range(3):
        sample = data[indexes[i]]
        ax = axes[i]
        img = sample['image']
        img = img[0]
        landmarks = sample['landmarks']
        landmarks = landmarks.reshape(-1, 2)
        gt_lms = sample['landmarks_truth']
        gt_lms = gt_lms.reshape(-1,2)
        ax.imshow(img,cmap='gray')
        ax.scatter(landmarks[:,0],landmarks[:,1],s=5,c='r')
        ax.scatter(gt_lms[:, 0], gt_lms[:, 1], s=5, c='g')
    plt.show()

# 5. 预测阶段
def predict(model, valid_loader,device):
    model.eval()
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

# 6. 微调阶段
def Finetune(args, train_loader, valid_loader, model, criterion, device):
    global loss
    print("Finetuning the models from checkpoint %s" % args.checkpoint)
    # 设定保存
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    # 设定训练次数、损失函数
    epoch = args.epochs
    criterion = criterion
    # 设置冻结层
    for para in list(model.parameters())[0:16]: #冻结IP2之上的参数
        para.requires_grad = False #取消自动求导
    # 设置只优化IP3 这一层 optimizer = optim.Adam(params=[model.ip3.weight, model.ip3.bias], lr=0.001, betas=(0.9, 0.99))
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    print("only the last layer -- IP3 -- will be trained, the other layers will be frozen")
    # monitor training loss
    train_losses = []
    valid_losses = []
    log_path = os.path.join(args.save_directory, 'log_info.txt')
    if (os.path.exists(log_path)):
        os.remove(log_path)
    # 训练
    for epoch_id in range(epoch):
        # training the model
        model.train()
        log_lines = []
        for batch_idx,batch in enumerate(train_loader):
            img = batch['image']
            input_img = img.to(device)
            # ground truth
            landmarks = batch['landmarks']
            target_pts = landmarks.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # get output
            output_pts = model(input_img)
            # get loss
            loss = criterion(output_pts,target_pts)
            # do BP automatically
            loss.backward()
            optimizer.step()
            # show log info
            if batch_idx % args.log_interval == 0:
                log_line = 'Train Epoch:{}[{}/{}({:.0f}%)]\t pts_loss:{:.6f}'.format(
                    epoch_id,
                    batch_idx * len(img), # 批次序号*一批次样本数=已测样本数
                    len(train_loader.dataset), # 总train_set样本数量
                    100.*batch_idx/len(train_loader), # 以上两者之比
                    loss.item()
                )
                print(log_line)
                log_lines.append(log_line)
        train_losses.append(loss)
        # 验证（使用测试数据集）
        valid_mean_pts_loss = 0.0
        model.eval()# prep model for evaluation
        with torch.no_grad():
            valid_batch_cnt = 0
            for batch_idx,batch in enumerate(valid_loader):
                valid_batch_cnt += 1
                valid_img = batch['image']
                input_img = valid_img.to(device)
                # ground truth
                landmarks = batch['landmarks']
                target_pts = landmarks.to(device)
                # result
                output_pts = model(input_img)
                valid_loss = criterion(output_pts,target_pts)
                valid_mean_pts_loss += valid_loss.item()
            #结论输出
            valid_mean_pts_loss /= valid_batch_cnt * 1.0
            log_line = 'Valid: pts_loss: {:.6f}'.format(valid_mean_pts_loss)
            print(log_line)
            log_lines.append(log_line)
            valid_losses.append(valid_mean_pts_loss)
        print('=============================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory,
                                            'detector_epoch'+'_'+str(epoch_id)+'.pt')
            torch.save(model.state_dict(), saved_model_name)
        # write log info
        with open(log_path, "a") as f:
            for line in log_lines:
                f.write(line + '\n')
    return train_losses,valid_losses

# 7. 运行
if __name__ == '__main__':
    main_test()
