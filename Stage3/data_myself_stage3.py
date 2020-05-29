import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools
import random
import cv2
import math

folder_list = ['I', 'II']
train_boarder = 112


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    classes = line_parts[-1]
    if classes == "1":
        landmarks = list(map(float, line_parts[5: len(line_parts) - 1]))
    else:
        landmarks = [0] * 42
    classes = np.long(classes)
    return img_name, rect, landmarks, classes


class Normalize(object):
    """
        Resize to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample["class"]
        w, h = image.size  # Image.size==>W*H

        image_resize = np.asarray(
            image.resize((train_boarder, train_boarder), Image.BILINEAR),
            dtype=np.float32)  # Image.ANTIALIAS)

        landmarks = np.float32(landmarks * [train_boarder / w, train_boarder / h])
        # image_resize = channel_norm(image_resize)
        landmarks = landmarks.flatten()

        return {'image': image_resize,
                'landmarks': landmarks,
                'class': classes
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample["class"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks),
                'class': classes
                }


class RandomFlip(object):
    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample["class"]
        # Flip image
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            landmarks[:, 0] = image.width - landmarks[:, 0]

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            landmarks[:, 1] = image.height - landmarks[:, 1]

        return {'image': image,
                'landmarks': landmarks,
                'class': classes
                }


class RandomRotate(object):
    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample["class"]
        # random rotate (-angle,angle)
        angle = 15
        a0 = random.uniform(-1, 1) * angle
        a1, a2 = a0, a0 * math.pi / 180
        ox, oy = image.width // 2, image.height // 2

        landmarks_f = landmarks.flatten()
        label = 0
        landmarks_new = [[ox + math.cos(a2) * (i - ox) - math.sin(a2) * (j - oy),
                          oy + math.sin(a2) * (i - ox) + math.cos(a2) * (j - oy)] for (i, j) in
                         zip(landmarks_f[::2], landmarks_f[1::2])]
        # 如果关键点坐标超过原图框，则不旋转
        for [i, j] in landmarks_new:
            if i > image.width or i < 0 or j > image.height or j < 0:
                label = 1
        if label == 0:
            image = image.rotate(-a1, Image.BILINEAR, expand=0)
            landmarks = np.array(landmarks_new).reshape(-1, 2)
        else:
            image = image
            landmarks = landmarks

        return {'image': image,
                'landmarks': landmarks,
                'class': classes
                }


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, transform=None):
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks, classes = parse_line(self.lines[idx])
        # image
        img = Image.open(img_name).convert('L')
        img_crop = img.crop(tuple(rect))
        if landmarks != "":
            landmarks = np.array(landmarks).reshape(-1, 2).astype(np.float32)

        sample = {'image': img_crop, 'landmarks': landmarks, "class": classes}
        sample = self.transform(sample)

        return sample


def load_data(phase):
    data_file = 'data\\' + phase + '.txt'

    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'stage_3_Train' or phase == 'stage_3_train':
        tsfm = transforms.Compose([
            # RandomFlip(),
            RandomRotate(),
            Normalize(),  # do channel normalization
            ToTensor()  # convert to torch type: NxCxHxW
        ])
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('stage_3_train')
    valid_set = load_data('stage_3_test')
    return train_set, valid_set


if __name__ == '__main__':
    train_set = load_data('stage_3_train')
    indexs = np.random.randint(0, len(train_set), 6)
    fig = plt.figure(figsize=(30, 10))
    axes = fig.subplots(nrows=1, ncols=6)
    for i in range(6):
        sample = train_set[indexs[i]]
        ax = axes[i]
        img = sample['image']
        img = img[0]
        landmarks = sample['landmarks']
        print(landmarks)
        ax.imshow(img, cmap='gray')
        ax.scatter(landmarks[::2], landmarks[1::2], s=5, c='r')
        # ax.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='r')
    plt.show()
