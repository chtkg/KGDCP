import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 读取CSV文件
data = pd.read_csv('./data/脑梗/train1.csv')

original_data_shape = data.shape
print("原始数据维度：", original_data_shape)


# 需要转换的列的名称列表
columns_to_convert = ['年龄', '糖化血红蛋白', '肌酐', '甘油三酯', '胆固醇', '高密度脂蛋白', '低密度脂蛋白', '脂蛋白a']

# 去除单位并将字符串转换为浮点数
for column in columns_to_convert:
    data[column] = data[column].str.replace(r'[^\d.]', '', regex=True).astype(float)

# 处理分类特征的独热编码
categorical_features = pd.get_dummies(data, columns=['性别', '口服药物', '合并症', '并发症'])

print(data['性别'].value_counts())
print(data['口服药物'].value_counts())
print(data['合并症'].value_counts())
print(data['并发症'].value_counts())


# 将布尔类型数据转换为整数（0或1）
categorical_features = categorical_features.astype(int)

# 将独热编码后的数据与数值特征合并
processed_data = pd.concat([data[columns_to_convert], categorical_features], axis=1)

# 转换为PyTorch张量
numerical_features = torch.tensor(processed_data[columns_to_convert].values, dtype=torch.float)
categorical_features = torch.tensor(categorical_features.values, dtype=torch.float)

print(numerical_features.shape[1])
print(categorical_features.shape[1])

# 打印数据的前几行，确保转换成功
print(processed_data.head())

# 初始化生成器和判别器
input_dim = numerical_features.shape[1] + categorical_features.shape[1]
output_dim = input_dim

print(output_dim)



# 定义生成器网络，输入维度为特征数量，输出维度为特征数量
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()  
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x



# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

generator = Generator(input_dim, output_dim)
discriminator = Discriminator(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

# 训练GANs的核心部分
num_epochs = 1000
batch_size = 64
real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    for i in range(0, numerical_features.size(0), batch_size):
        # 随机抽取一批真实数据
        real_data = torch.cat((numerical_features[i:i+batch_size], categorical_features[i:i+batch_size]), dim=1)
        
        # 生成合成数据
        noise = torch.randn(batch_size, input_dim)
        synthetic_data = generator(noise)
        
        # 训练判别器
        optimizer_D.zero_grad()
        #real_output = discriminator(real_data)
        real_label = 1.0

        real_output = discriminator(real_data).float()

        fake_output = discriminator(synthetic_data)
        real_loss = criterion(real_output, torch.full((real_data.size(0), 1), real_label, dtype=torch.float))
        fake_loss = criterion(fake_output, torch.full((batch_size, 1), fake_label, dtype=torch.float))


        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # 训练生成器
        optimizer_G.zero_grad()
        noise = torch.randn(batch_size, input_dim)
        synthetic_data = generator(noise)
        fake_output = discriminator(synthetic_data)
        g_loss = criterion(fake_output, torch.full((batch_size, 1), real_label))
        g_loss.backward()
        optimizer_G.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}')

# 生成合成数据，这里使用生成器来生成合成数据
num_synthetic_samples = 1000
noise = torch.randn(num_synthetic_samples, input_dim)
synthetic_data = generator(noise)



# 创建一个只包含生成的合成数据列名的列表
synthetic_column_names = [f'合成特征_{i}' for i in range(1, output_dim + 1)]

# 创建 DataFrame
synthetic_df = pd.DataFrame(synthetic_data.cpu().detach().numpy(), columns=synthetic_column_names)

# 保存合成数据到 CSV 文件
synthetic_df.to_csv('./data/脑梗/synthetic_data.csv', index=False)


