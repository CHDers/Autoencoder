# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/3 16:08
# @Author  : Yanjun Hao
# @Site    : 
# @File    : autoencoder.py
# @Software: PyCharm 
# @Comment :


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
from sklearn import datasets
from rich import print


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 定义自定义数据集
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return torch.Tensor(x)


def main():
    # SECTION: 读取数据
    dateset_ = datasets.load_breast_cancer()
    print(dateset_.keys())
    data = dateset_["data"]
    target = dateset_["target"]
    df = pd.DataFrame(data, columns=dateset_["feature_names"])
    df["target"] = target

    print(df.shape)

    feature_select = dateset_["feature_names"]

    # 提取特征和标签
    x_data, y_data = df[feature_select], df["target"]

    # SECTION：参数初始化
    input_size = x_data.shape[1]  # 输入数据维度
    hidden_size = 64  # 隐藏层维度
    output_size = 6  # 输出维度
    learning_rate = 0.001  # 学习率
    num_epochs = 200  # 迭代次数
    batch_size = 16  # 每次迭代的批次大小

    # 数据标准化
    standardScaler = preprocessing.StandardScaler()
    standardScaler.fit(x_data)
    x_train = standardScaler.transform(x_data)

    # 加载数据集
    train_dataset = MyDataset(x_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型初始化
    autoencoder = Autoencoder(input_size, hidden_size, output_size)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in tqdm(range(num_epochs), desc="Train", colour="green"):
        for data in train_loader:
            inputs = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # 测试模型
    autoencoder.eval()
    with torch.no_grad():
        features = []
        for data in train_loader:
            feature = autoencoder.encoder(data)
            features.append(feature)
        features = torch.cat(features, dim=0)
    features = features.numpy()
    print(features.shape)
    features = pd.DataFrame(features)
    features.to_csv("AE_features.csv", index=False)


if __name__ == '__main__':
    main()
