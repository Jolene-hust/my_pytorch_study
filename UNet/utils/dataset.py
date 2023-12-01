import torch
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 参数是文件夹的路径
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, "image/*.png"))

    
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，flipCode为1水平翻转，0垂直翻转，-1水平加垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转化为灰度图，并reshape为三维
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1，因为label的像素点是0或1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    
    def __len__(self):
        return len(self.imgs_path)



if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("数据个数: ", len(isbi_dataset))
    train_loader = DataLoader(isbi_dataset, batch_size=2, shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        