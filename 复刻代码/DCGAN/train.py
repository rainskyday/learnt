import torch
import torchvision
import torch.nn as nn
import numpy as np
import os

# MNIST数据集 单通道28*28
image_size = [1, 28, 28]
# 潜在变量 噪声大小
z_dim = 100
# 批训练大小
batch_size = 16
# 设备选取
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1、准备数据集
dataset = torchvision.datasets.MNIST(root='../data/mnist', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                         ]
                                     ))
if not os.path.exists("results"):
    os.makedirs("results")

# 2、加载数据集
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 3、搭建神经网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 网络结构
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 4*4*512),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            # input: 4 by 4, output: 7 by 7
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # input: 7 by 7, output: 14 by 14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # input: 14 by 14, output: 28 by 28
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)
        y_ = self.fc(x)
        y_ = y_.view(y_.size(0), 512, 4, 4)
        y_ = self.conv(y_)
        return y_

# 判别器
class Discriminator(nn.Module):
    """
        Convolutional Discriminator for MNIST
    """
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # 28 -> 14
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 14 -> 7
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 7 -> 4
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Sequential(
            # reshape input, 128 -> 1
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        y_ = self.conv(x)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_


# 4、创建网络模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 5、设置损失函数、优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss_fn = nn.BCEWithLogitsLoss()
labels_one = torch.ones(batch_size, 1).to(device)
labels_zero = torch.zeros(batch_size, 1).to(device)

# 6、训练网络
num_epoch = 50
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_images = mini_batch[0].to(device)

        z = torch.randn(batch_size, z_dim).to(device)

        fake_images = generator(z)

        # 生成器损失
        g_loss = loss_fn(discriminator(fake_images), labels_one)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # 判别器损失
        real_loss = loss_fn(discriminator(gt_images), labels_one)
        fake_loss = loss_fn(discriminator(fake_images.detach()), labels_zero)
        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        if i % 500 == 0:
            print(f"Epoch:{epoch}, step:{len(dataloader) * epoch + i}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}")
        if i % 7500 == 0:
            image = fake_images[:16].data
            torchvision.utils.save_image(image, f"results/image_{len(dataloader) * epoch + i}.png", nrow=4)
