# 引入库文件
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
# Train
# 数据集载入，这里是直接使用官方的MNIST数据集，存放在mnist_data路径下
dataset = torchvision.datasets.MNIST(root='../data/mnist', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                         ]
                                     ))
# 结果存储
if not os.path.exists("results"):
    os.makedirs("results")

# 2、加载数据集
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 3、搭建神经网络
# 搭建生成器
# 生成器的输入是一个100维的随机噪声，输出是一个784维的向量，代表一张28*28的图片
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),

            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),
            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),
            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),
            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # shape of z: [batchsize, latent_dim]
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)
        return image

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype=np.int32), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        # shape of image: [batchsize, 1, 28, 28]
        prob = self.model(image.reshape(image.shape[0], -1))
        return prob

# 4、创建网络模型
generator = Generator()
generator.to(device)
discriminator = Discriminator()
discriminator.to(device)


# 5、设置损失函数、优化器灯参数
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1).to(device)
labels_zero = torch.zeros(batch_size, 1).to(device)

# 6、训练网络
num_epoch = 50
for epoch in range(num_epoch):
    print(f"Epoch:{epoch}")
    print(len(dataloader))
    for i, mini_batch in enumerate(dataloader):

        gt_images = mini_batch[0].to(device)

        z = torch.randn(batch_size, z_dim).to(device)

        fake_images = generator(z).to(device)

        g_loss = loss_fn(discriminator(fake_images), labels_one).to(device)
        # 梯度清零
        g_optimizer.zero_grad()
        # 反向传播
        g_loss.backward()
        # 更新梯度方向
        g_optimizer.step()

        # 判别器loss
        real_loss = loss_fn(discriminator(gt_images), labels_one).to(device)
        fake_loss = loss_fn(discriminator(fake_images.detach()), labels_zero).to(device)
        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 7、获取模型结果
        if i % 500 == 0:
            # print(f"step:{len(dataloader)*epoch+i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")
            print(f"step:{len(dataloader) * epoch + i}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")
        if i % 7500 == 0:
            image = fake_images[:16].data
            torchvision.utils.save_image(image, f"results\image_{len(dataloader) * epoch + i}.png", nrow=4)
