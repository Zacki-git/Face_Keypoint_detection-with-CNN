import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# 1. 设定数据预处理标准
train_boarder = 112

## 1.1 Normalize
def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean)/(std+0.0000001)
    return pixels

class Normalize(object):
    def __call__(self,sample):
        image,landmarks = sample['image'],sample['landmarks']
        w,h = image.size # Image.size==>W*H
        # 先对image进行resize至train_boarder*train_boarder
        image_resize = np.asarray(image.resize((train_boarder,train_boarder),Image.BILINEAR),
                                 dtype = np.float32)
        # 同时应对landmarks进行相应调整
        landmarks = np.float32(landmarks * [train_boarder/w,train_boarder/h])
        landmarks = landmarks.reshape(-1)
        # 再对image进行norm
        image = channel_norm(image_resize)
        return{'image':image, 'landmarks':landmarks}

## 1.2 ToTensor
class ToTensor(object):
    # Convert ndarrays in sample to Tensors.Tensors channel sequence: N*C*H*W
    def __call__(self,sample):
        image,landmarks = sample['image'],sample['landmarks']
        # numpy image:H*W*C
        # torch image:C*H*W
        # 理论上要进行image = image.transpose((2,0,1))
        # 由于image读入‘L’mode，所以不需要转换
        image = np.expand_dims(image,axis=0)
        return {'image':torch.from_numpy(image),
                'landmarks':torch.from_numpy(landmarks)}

# 2. 读取并预处理数据
## 2.1 FaceLandmarksDataset
def parse_line(line):
    lines = line.strip().split()
    name = lines[0]
    rect = list(map(int,list(lines[1:5])))
    landmarks = list(map(float,lines[5:]))
    return name,rect,landmarks

class FaceLandmarksDataset(Dataset):
    # 读取原始数据，转化为可预处理的字典形式
    def __init__(self, lines, transform=None):
        self.lines = lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        name, rect, landmarks = parse_line(self.lines[idx])
        # 对图像进行裁剪，成为人脸框图片
        img = Image.open(name).convert('L')
        img_crop = img.crop(tuple(rect))
        landmarks = np.array(landmarks).reshape(-1, 2).astype(np.float32)
        # 对landmarks的处理在Normalize类中
        # 个人觉得图像和关键点预处理步骤应同步，避免不同transform方法时造成问题
        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)
        return sample

## 2.2 load_data
def load_data(phase):
    file = 'data\\' + phase + '.txt'
    with open(file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        transform = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    else: # 其它阶段可考虑改变预处理方式
        transform = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data = FaceLandmarksDataset(lines,transform=transform)
    return data

## 2.3 get_train_test_set()
def get_train_test_set():
    train_set = load_data('train')
    test_set = load_data('test')
    return train_set,test_set

# 3. 检验数据正确性
if __name__ == '__main__':
    train_set = load_data('train')
    indexs = np.random.randint(0,len(train_set),3)
    fig = plt.figure(figsize=(10,10))
    axes = fig.subplots(nrows=1,ncols=3)
    for i in range(3):
        sample = train_set[indexs[i]]
        ax = axes[i]
        img = sample['image']
        img = img[0]
        landmarks = sample['landmarks']
        landmarks = landmarks.reshape(-1, 2)
        ax.imshow(img,cmap='gray')
        ax.scatter(landmarks[:,0],landmarks[:,1],s=5,c='r')
    plt.show()



