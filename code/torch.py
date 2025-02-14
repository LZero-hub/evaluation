import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dataloader import DataLoader
from dataloader import pick_data, pick_data_more
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载数据
data_dir = 'other_data_feature29'
file_extension = '.npy'
data_loader = DataLoader(data_dir, file_extension)
data = data_loader.load_data()
X = np.array(data)
print(X.shape)
X = X.reshape((X.shape[0], -1))
print(X.shape)
y = np.load('scorenew25.npy').astype(float)

# 准备数据集
x_train, x_val, y_train, y_val = pick_data(X, y, 0.7, 0.3, 3, 500)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).float()

# 使用 DataLoader 创建批量数据加载器
batch_size = 16  # 设置新的 batch_size
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
val_dataset = torch.utils.data.TensorDataset(x_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class R2Score:
    def __init__(self):
        self.y_pred = torch.tensor([]).to(device)
        self.y_true = torch.tensor([]).to(device)

    def update(self, y_true, y_pred):
        self.y_pred = torch.cat((self.y_pred, y_pred), dim=0).to(device)
        self.y_true = torch.cat((self.y_true, y_true), dim=0).to(device)

    def compute(self):
        mean_true = torch.mean(self.y_true)
        total_sum_of_squares = torch.sum((self.y_true - mean_true) ** 2).to(device)
        residual_sum_of_squares = torch.sum((self.y_true - self.y_pred) ** 2).to(device)
        r2_score = 1 - residual_sum_of_squares / total_sum_of_squares
        return r2_score.item()


class Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(p=0)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(p=0)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available")
else:
    device = torch.device("cpu")
    print("cuda is not available")

flag = False
a = input("是否重新训练Y/N")
if a == 'Y':
    flag = True
else:
    device = torch.device("cpu")

if flag:
    start_time = time.time()
    input_dim = 4000
    hidden_dim1 = 1024
    hidden_dim2 = 256
    hidden_dim3 = 128
    hidden_dim4 = 8
    hidden_dim5 = 4
    output_dim = 1
    lambda_value = 0.0001

    model = Regressor(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000058)

    num_epochs = 2000
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        r2_score = R2Score()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            l2_regularization = sum(torch.norm(param, 2) for param in model.parameters()).to(device)
            loss += lambda_value * l2_regularization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            r2_score.update(targets, outputs)

        avg_loss = running_loss / len(train_loader)
        epoch_r2 = r2_score.compute()

        model.eval()
        val_r2 = R2Score()
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_r2.update(val_targets, val_outputs)

        epoch_val_r2 = val_r2.compute()

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train R^2: {epoch_r2:.4f}, Val R^2: {epoch_val_r2:.4f}')

        '''if epoch_r2 > 0.9:
            torch.save(model, 'model11.pth')
            break'''

    torch.save(model, 'model17.pth')
    train_time = time.time() - start_time
    print("train_time:", train_time)

model = torch.load('model17.pth').to("cpu")
'''for name, param in model.named_parameters():
    print(f"Layer: {name} | Weights: {param.data}")
    weights = param.data.cpu().numpy()

    # 创建序列索引
    indices = range(weights.size)

    # 绘制权重图
    plt.figure(figsize=(10, 5))
    plt.plot(indices, weights.flatten(), marker='o')  # 横坐标为索引，纵坐标为权重值
    plt.title(f"Weights of {name}")
    plt.xlabel('Index')
    plt.ylabel('Weight Value')
    plt.grid()
    plt.show()'''
test_data = torch.from_numpy(X[0:]).float().to(device)

predicted = model(test_data) #0.125

y1 = [y[i+0] / 10 for i in range(len(y[0:]))]
y1_array = np.array(y1)
tensor_y = torch.Tensor(y1_array).to(device)

r2_score = R2Score()
r2_score.update(tensor_y, predicted)
print("R^2:", r2_score.compute())

predicted = predicted.to("cpu").detach().numpy()
result=np.array2string(predicted)
with open("torch_result_1.txt",'a')as file:
    for i in range(len(predicted)):
        #file.write(str(y1[i][0]))
        #file.write(" ")
        file.write(str(predicted[i][0]))
        file.write('\n')
#evaluate_time = time.time() - train_time - start_time
#print("evaluate_time:", evaluate_time)
#plt.scatter(tensor_y,predicted)
plt.plot(range(len(predicted)), predicted, label='predict')
plt.plot(range(len(predicted)), tensor_y, label='real')
plt.scatter(range(len(predicted)), predicted, label='predict')
plt.scatter(range(len(predicted)), tensor_y, label='real')
plt.legend()
plt.show()

