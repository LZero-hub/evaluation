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
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available")
else:
    device = torch.device("cpu")
    print("cuda is not available")

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
if __name__=='__main__':
    data_dir = 'other_data_feature27'
    file_extension = '.npy'
    data_loader = DataLoader(data_dir, file_extension)
    data = data_loader.load_data()
    X = np.array(data)
    print(X.shape)
    X = X.reshape((X.shape[0], -1))
    print(X.shape)
    model=torch.load("model17.pth").to(device)
    test_data = torch.from_numpy(X[0:1499]).float().to(device)
    predicted = model(test_data)  # 0.125
    min_value_index=torch.argmin(predicted)
    max_value_index=torch.argmax(predicted)
    value_index=min_value_index
    print(value_index)
    input_test=[]
    k_list=[]
    output_origin=model(test_data[value_index])
    output_origin_numpy=output_origin.to("cpu").detach().numpy()
    test_result=test_data[value_index].to("cpu").detach().numpy()
    #plt.plot(range(len(test_result)),test_result,label='look')
    #plt.show()
    print(output_origin)
    print(test_result)
    path='location1'
    propotion=0.001
    n=0
    for j in range(4):
        input_test = []
        k_list = []
        max_output=-1
        min_output=1
        for i in range(1000):
            input_data=test_data
            #print(test_data[min_value_index])
            input_data[value_index][j*1000+i]=test_data[value_index][j*1000+i]+torch.abs(test_data[value_index][j*1000+i]*propotion)
            output=model(input_data[value_index])
            derta_x=(test_data[value_index][j*1000+i]*propotion)
            derta_y=(output-output_origin)/output_origin
            if derta_y>max_output:
                max_output=derta_y
            if derta_y<min_output:
                min_output=derta_y
            derta_x=np.array(derta_x.to("cpu").detach().numpy())
            derta_y=np.array(derta_y.to("cpu").detach().numpy())
            #print(derta_x)
            k=derta_y/np.abs(derta_x)
            input_test.append(derta_y[0])
            k_list.append(k)
        #print(k_list)
        #print(input_test)
        path1=path+'/text.txt'
       # path1=path+'/text{}.txt'.format(j)
        print((input_test))
        top_50_indices = np.argsort(input_test)[-50:]  # 找到最大的 50 个索引
        print(top_50_indices)
        n = n + 1
        for i in range(50):
            index = top_50_indices[i]  # 当前最大值的索引
            if not isinstance(index, int):
                index = int(index)  # 强制转换为整数
            # 计算对应的 x, y, z 值
            x = (index // 100) * 2
            y = (index % 100 // 10) * 2
            z = index % 10 * 2

            # 获取对应的 input_test 值，并修改其值
            value = np.abs(input_test[index] * 100)

            # 将 x, y, z, i 和对应的 input_test 值写入文件
            with open(path1, 'a') as f:
                f.write('{} {} {} {} {} \n'.format(x, y, z,n, value))
        #plt.plot(range(len(k_list)),k_list, label='k')
        plt.plot(range(len(input_test)),input_test,label='derta_y')
        plt.show()
        #print(k_list.index(max(k_list)))
        ''' path1=path+'/result{}.txt'.format(j)
               for i in range(1000):
                   with open(path1,'a')as f:
                       f.write('{} {} \n'.format(i, input_test[i][0]))'''