import os

import numpy as np

class DataLoader:
    def __init__(self, data_dir, file_extension='.npy'):
        self.data_dir = data_dir
        self.file_extension = file_extension
        self.file_paths = self._get_file_paths()

    def _get_file_paths(self):
        file_paths = os.listdir(self.data_dir)
        return file_paths

    def load_data(self):
        data = []
        for file_path in self.file_paths[0:10000]:
            #print(file_path)
            file_path=os.path.join(self.data_dir,file_path)
            loaded_data = np.load(file_path)
            #print(loaded_data)
            #print(loaded_data.shape)
            #pca的场景单个数据
            #loaded_data = loaded_data.reshape(loaded_data.shape[0], -1)
            #print(loaded_data.shape)
            #loaded_data=loaded_data.reshape(loaded_data.shape[0]*loaded_data.shape[1],loaded_data.shape[2])
            #encoder的场景reshape
            #loaded_data = loaded_data.reshape(-1)
            data.append(loaded_data)
            # Add more conditionals for other file formats if needed
        return data
def pick_data(x,y,proportion=0.7,test_proprtion=0.3,data_kind_num=0,num=0):
    data_len=int(num*proportion)
    test_data_len=int(num*test_proprtion)
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    for i in range(0,data_kind_num):
        x_train.append(x[i*num:i*num+data_len])
        y_train.append(y[i*num:i*num+data_len]/10)
        x_val.append(x[i*num+data_len:(i)*num+data_len+test_data_len])
        y_val.append(y[i * num + data_len:i * num+data_len+test_data_len]/10)
    return x_train,x_val,y_train,y_val
def pick_data_more(x,y,obstacle_num=1,proportion=0.7,test_proprtion=0.3,data_kind_num=0,num=0):
    data_len=int(num*proportion)
    test_data_len=int(num*test_proprtion)
    total=int(data_kind_num*num)
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    for j in range(obstacle_num):
        for i in range(data_kind_num):
            x_train.append(x[j*total+i*num:j*total+i*num+data_len])
            y_train.append(y[j*total+i*num:j*total+i*num+data_len]/10)
            x_val.append(x[j*total+i*num+data_len:j*total+(i)*num+data_len+test_data_len])
            y_val.append(y[j*total+i * num + data_len:j*total+i * num+data_len+test_data_len]/10)
    return x_train,x_val,y_train,y_val
if __name__=='__main__':
    data_dir = 'other_data_feature4'
    file_extension = '.npy'
    data_loader = DataLoader(data_dir, file_extension)
    data = data_loader.load_data()
    data=np.array(data)
    print(data.shape)
