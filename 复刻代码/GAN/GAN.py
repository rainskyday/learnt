import torch
import torch.nn as nn
import torchvision.datasets

image_size = [1, 28, 28]
image_size_prod = torch.prod(torch.tensor(image_size))
# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size_prod, 64),
            torch.nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            torch.nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            torch.nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            torch.nn.ReLU(inplace=True),
            nn.Linear(1024, image_size_prod),
            nn.Tanh(),
        )

    def forward(self, z):
        # shape of z: (batch_size, 1, 28, 28)
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)
        return image



# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size_prod, 1024),
            torch.nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, image):
        # shape of image: (batch_size, 1, 28, 28)
        prob = self.model(image.reshape(image.shape[0], -1))
        return prob


# train
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                     transform=torchvision.transforms.Compose(
                                         [
                                             torchvision.transforms.Resize(28),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
                                         ]
                                                                            )
                                     )
# print(len(dataset))
# for i in range(len(dataset)):
#     if i < 5:
#         print(dataset[i][0].shape)
#     else:
#         break

batch_size = 32

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


generator = Generator()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
discriminator = Discriminator()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()

num_epoch = 50
latent_dim = image_size_prod
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_images, _ = mini_batch
        z = torch.randn(batch_size, latent_dim)
        pred_image = generator(z)

        # train generator
        g_optimizer.zero_grad()
        target = torch.ones(batch_size, 1)
        g_loss = loss_fn(discriminator(pred_image), target)
        g_loss.backward()
        g_optimizer.step()

        # train discriminator
        d_optimizer.zero_grad()
        d_loss = 0.5 * loss_fn(discriminator(gt_images), target) + 0.5 * loss_fn(discriminator(pred_image.detach()), 1-target)
        d_loss.backward()
        d_optimizer.step()

        if i % 1000 == 0:
            for index, image in enumerate(pred_image):
                torchvision.utils.save_image(pred_image, f'./result/{epoch}_{index}.png', nrow=8, normalize=True)


