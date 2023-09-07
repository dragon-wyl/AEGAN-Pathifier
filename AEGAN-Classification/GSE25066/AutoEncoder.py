# %% md
# 使用AutoEncoder降维
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


# 定义 AutoEncoder 网络
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 定义训练函数
def train(model, dataloader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        # 打印损失
        if epoch % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))


# %%
def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor, min_val, max_val


def min_max_denormalize(normalized_tensor, min_val, max_val):
    denormalized_tensor = normalized_tensor * (max_val - min_val) + min_val
    return denormalized_tensor


# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# %%
setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
# %%
# 加载数据
data = pd.read_csv('GSE25066_merge.csv')
data = data[data['group'] == 0]
data.pop('group')
data_tensor = torch.from_numpy(data.values).to(torch.float32).to(device)
nor_data_tensor, ae_min, ae_max = min_max_normalize(data_tensor)
print(nor_data_tensor)
dataset = TensorDataset(nor_data_tensor, nor_data_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 创建 AutoEncoder 模型
input_dim = 13236
encoding_dim = 512  # 降维后的维度
model = AutoEncoder(input_dim, encoding_dim).to(device)

# 训练模型
num_epochs = 10
learning_rate = 0.0001

# %%
dataset
# %%
model.load_state_dict(torch.load('GSE25066_AE.pth'))
# %%
# train(model, dataloader, 1, learning_rate)
# %%
# 使用训练好的模型对数据进行降维
encoded_data = model.encoder(nor_data_tensor)
# %%
nor_data_tensor
# %%
data_tensor
# %%
encoded_data
# %%
model.decoder(encoded_data)
# %%
min_max_denormalize(model.decoder(encoded_data), ae_min, ae_max)
# %%
torch.save(model.state_dict(), 'GSE25066_AE.pth')
# %% md
# 生成对抗网络
# %%
import torch
import torch.nn as nn
import torch.optim as optim


# 定义生成器（Generator）
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# 定义判别器（Discriminator）
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 超参数设置
input_dim = 64
data_dim = 512
lr = 0.0002
epochs = 1000
batch_size = 64

# 初始化生成器和判别器
generator = Generator(input_dim, data_dim).to(device)
discriminator = Discriminator(data_dim).to(device)

# 设置优化器
g_optimizer = optim.Adam(generator.parameters(), lr=lr)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# 设置损失函数
loss_func = nn.BCELoss()

# %%
encoded_data.shape
# %%
gan_dataset = TensorDataset(encoded_data, encoded_data)
gan_dataloader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True)
# %%
# 训练GAN
discriminator.train()
generator.train()
min_loss = 1000
for epoch in range(epochs):
    for data in gan_dataloader:
        # print("********")
        real_data, _ = data
        real_data = real_data.to(device)
        # 训练判别器
        d_optimizer.zero_grad()
        real_label = torch.ones(real_data.shape[0], 1).to(device)

        fake_data = generator(torch.randn(real_data.shape[0], input_dim).to(device)).detach()
        fake_label = torch.zeros(real_data.shape[0], 1).to(device)
        real_out = discriminator(real_data)
        fake_out = discriminator(fake_data)
        real_loss = loss_func(real_out, real_label)
        fake_loss = loss_func(fake_out, fake_label)
        d_loss = real_loss + fake_loss
        d_loss.backward(retain_graph=True)
        d_optimizer.step()

        # 训练生成器
        g_optimizer.zero_grad()
        gen_input = torch.randn(real_data.shape[0], input_dim).to(device)
        gen_output = generator(gen_input)
        dis_output = discriminator(gen_output)

        g_loss = loss_func(dis_output, real_label)

        g_loss.backward()
        g_optimizer.step()
        if g_loss.item() < min_loss:
            torch.save(generator.state_dict(), "generator_0.pth")
    if epoch % 10 == 0:
        print("Epoch: {}, G_Loss: {:.4f}, D_Loss: {:.4f}".format(epoch, g_loss.item(), d_loss.item()))

# %%
