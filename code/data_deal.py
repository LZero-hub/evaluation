import numpy as np
import os
import time
from scipy.signal import convolve
#原始版
def calculate_score(data_vector):
    p_num=[23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,91,101,103,107]
    data_num=[0 for _ in range(20)]
    score=0
    for i in range(data_vector.shape[0]):
        for j in range(data_vector.shape[1]):
            for k in range(data_vector.shape[2]):
                if data_vector[i,j,k]>0:
                    data_num[int(data_vector[i,j,k])-1]+=1
                    score+=1/p_num[int(data_vector[i,j,k])-1]
    return score
def numtest(data_vector):
    total_num = 27
    # kernel = np.ones((3, 3, 3))
    # output_array = convolve(data_vector, kernel, mode='valid')
    output_array = []
    for i in range(0, 20, 3):
        for j in range(0, 20, 3):
            for k in range(0, 20, 3):
                flag = False
                path_score = 0
                data_distance_score = 0
                data_deal = data_vector[i:i + 4, j:j + 4, k:k + 4]
                data_sum = np.sum(data_deal)
                data_abs = np.sum(np.abs(data_deal))
                if (data_sum > 0) or (abs(data_sum) != data_abs):
                    flag = True
                if not flag:  # 障碍物区域
                    path_score = data_sum / total_num
                else:
                    '''长度评分相关'''
                    num_path = (data_abs + data_sum) / 2
                    num_obstacle = (data_sum - data_abs) / 2
                    distance_start = np.sqrt((i + 1) ** 2 + (j + 1) ** 2 + (k + 1) ** 2) / 20
                    #data_distance_score = distance_start * num_path / total_num
                    data_distance_score=calculate_score(data_deal)
                    # output_array[i,j,k]=data_distance_score
                    '''关于与障碍物距离评分'''
                    path_score = data_sum / total_num
                # print([path_score,data_distance_score])
                output_array.append([path_score, data_distance_score])
    return output_array
def judgement(data_vector,path_start,path_aim):
    distance_short=[]
    #for i in range(len(path_start)):
     #   distance_short.append(np.sqrt((path_start[i][0]-path_aim[i][0])**2+(path_start[i][1]-path_aim[i][1])**2+(path_start[i][2]-path_aim[i][2])**2))

   #kernel = np.ones((3, 3, 3))
    #output_array = convolve(data_vector, kernel, mode='valid')
    output_array_1 = []
    output_array_2 = []
    z=3
    total_num = (z+1)**3
    for i in range(0,20,z):
        for j in range(0,20,z):
            for k in range(0,20,z):
                flag=False
                path_score=0
                data_distance_score=0
                data_deal=data_vector[i:i+z+1,j:j+z+1,k:k+z+1]
                data_sum=np.sum(data_deal)
                data_abs=np.sum(np.abs(data_deal))
                if (data_sum>0) or (abs(data_sum)!=data_abs):
                    flag=True
                if not flag:#障碍物区域
                    path_score=data_sum/total_num
                else:
                    '''长度评分相关'''
                    num_path=(data_abs+data_sum)/2
                    num_obstacle=(data_sum-data_abs)/2
                    distance_start=np.sqrt((i+1)**2+(j+1)**2+(k+1)**2)/20
                    data_distance_score=calculate_score(data_deal)
                    #output_array[i,j,k]=data_distance_score
                    '''关于与障碍物距离评分'''
                    path_score=data_sum/total_num
                #print([path_score,data_distance_score])
                output_array_1.append(path_score)
                output_array_2.append(data_distance_score)
    return output_array_1,output_array_2


def connection_judgement(data_vector_connection,path_start,path_aim):
    output_array_2= []
    output_array_3=[]
    z=3
    for i in range(0, 20, z):
        for j in range(0, 20, z):
            for k in range(0, 20, z):
                flag = False
                connect_angle_score = 0
                # print(i,j,k)
                '''if (k==16):
                    data_deal=data_vector_connection[i:i+5,j:j+5,k-1:k+4]
                if (j==16):
                    data_deal=data_vector_connection[i:i+5,j-1:j+4,k:k+5]
                if (i==16):
                    data_deal = data_vector_connection[i-1:i + 4, j :j + 5, k:k + 5]
                else:'''
                data_deal = data_vector_connection[i:i + z+1, j:j + z+1, k:k + z+1]
                # print(data_deal.shape)
                angle_array,distance_array= lookfor(data_deal)
                output_array_2.append(angle_array[0])
                output_array_3.append(distance_array[0])
    return output_array_2,output_array_3
def lookfor(data_vector):
    '''通过对所在点的六点判断去考虑起是否是起终点，起终点应该只有一个方向存在与其标记一样的空间点'''
    path_start=[[]for _ in range(20)]
    path_aim=[[]for _ in range(20)]
    #print(data_vector)
    dict=[[0,0,1],[0,1,0],[1,0,0],[-1,0,0],[0,-1,0],[0,0,-1]]
    for i in range(data_vector.shape[0]):
        for j in range(data_vector.shape[1]):
            for k in range(data_vector.shape[2]):
                num=0
#                if path_start[int(data_vector[i,j,k])]==[]:
#                   path_start[int(data_vector[i,j,k])]=[i,j,k]
                new_list=[[i+elem[0],j+elem[1],k+elem[2]]for elem in dict]
                if (int(data_vector[i,j,k])>0):
                    #print(new_list)
                    for elem in new_list:
                        #print(min(data_vector.shape[0],data_vector.shape[1],data_vector.shape[2]))
                        if (elem[0]<0)or (elem[1]<0)or(elem[2]<0):
                            continue
                        if ((elem[0]>data_vector.shape[0]-1) or (elem[1]>data_vector.shape[1]-1)
                                or(elem[2]>(data_vector.shape[2]-1))):
                            continue
                        #print(elem[0],elem[1],elem[2])
                        #print(data_vector.shape)
                        #print("elem:",[elem[0], elem[1], elem[2]], data_vector[elem[0], elem[1], elem[2]])
                        #print("origin:",[i, j, k], data_vector[i, j, k])
                        if data_vector[elem[0],elem[1],elem[2]]==data_vector[i,j,k]:
                            num+=1
                        #print(num)
                    if num<=1:
                        #print(data_vector[i,j,k])
                        #print(path_aim[int(data_vector[i,j,k])])
                        path_aim[int(data_vector[i,j,k])-1].append([i,j,k])
                        #num=0
    angle_array=[]
    distance_array=[]
    #print(path_aim)
    distance=[]
    vector=[]
    for elem in path_aim:
        if len(elem)==2:
            vector.append([elem[1][0]-elem[0][0],elem[1][1]-elem[0][1],elem[1][2]-elem[0][2]])
            distance.append([(elem[1][0]+elem[0][0])/2,(elem[1][1]+elem[0][1])/2,(elem[1][2]+elem[0][2])/2])
        if len(elem)==1:
            #vector.append(elem[0])
            #distance.append(elem[0])
            continue
    #print(vector)
    angle_array.append(angle_output(vector))
    distance_array.append(distance_output(distance))
    return [angle_array,distance_array]
def distance_output(vectors):
    distance=0
    min_distance_product = float('inf')
    length = len(vectors)
    # print(length)
    # 对于一块区域的单点部分，还需要进一步修改
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if (vectors[i] == []) or (vectors[j] == []):
                continue
            vector=[vectors[i][0]-vectors[j][0],vectors[i][1]-vectors[j][1],vectors[i][2]-vectors[j][2]]
            distance=np.sqrt(np.sum(np.square(vector)))
            #print(distance)
            if distance<min_distance_product:
                min_distance_product=distance
            # print(vectors[i])
            # print(vectors[j])
    if min_distance_product==float('inf'):
        min_distance_product=0
    return min_distance_product
def angle_output(vectors):
    min_dot_product = float(10)
    max_dot_product=float(-10)
    # 遍历每对向量，并计算它们的点积
    length=len(vectors)
    #print(length)
    cosnum=0
    number=0
    #对于一块区域的单点部分，还需要进一步修改
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if (vectors[i]==[])or(vectors[j]==[]):
                continue
            #print(vectors[i])
            #print(vectors[j])
            number+=1
            dot_product = np.dot(vectors[i], vectors[j])
            cosnum+=np.abs(dot_product/(np.sqrt(np.sum(np.square(vectors[i]))))/np.sqrt(np.sum(np.square(vectors[j]))))
            #print(dot_product,cosnum)
    if number==0:
        return -1
    else:
        average_dot_product=cosnum/number
    return average_dot_product
def data_deal_std(X):
    X = X.reshape((X.shape[0], -1))
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    zero_std_indices = np.where(std == 0)[0]
    std[zero_std_indices] = 1
    # 标准化数据
    X = (X - mean) / std
    return X
total_file_path='data202409132'
file_list=os.listdir(total_file_path)
n = 1000
score=[]
start_time=time.time()
for file_name in file_list:
    file_obstacle=os.path.join(total_file_path,file_name,'obstacleData','ObstacleData.txt')
    file_path=os.path.join(total_file_path,file_name,'pathData')
    file_path_list=os.listdir(file_path)
    path_data=[]
    obstacle_data=[]
    with open(file_obstacle,'r') as file:
        lines = file.readlines()
        # 处理每一行的数据
        for line in lines[1:]:
            # 对每一行进行处理，例如打印内容
            x, y, z, f = line.strip().split()
            obstacle_data.append([int(x), int(y), int(z), int(f)-1])
    for data_path in file_path_list:
        data_path=os.path.join(file_path,data_path)
        path_start = []
        path_aim = []
        print(data_path)
        with open(data_path,'r') as file:
            # 逐行读取数据
            lines = file.readlines()
            # 处理每一行的数据
            score1,score2,score3,score4=lines[0].strip().split()
            #score0 = (float(score3) + float(score4)) / 2
            score0=(float(score1)+float(score2)+float(score3)+float(score4))/4
            score.append([score0])
            for line in lines[1:]:
                # 对每一行进行处理，例如打印内容
                x, y, z, f = line.strip().split()
                flag1=int(f) % 3
                flag2=int(f) // 3+1
                if (flag1==1):
                    path_data.append([int(x), int(y), int(z), flag2])
                if flag1==2:
                    path_start.append([int(x), int(y), int(z)])
                if flag1==3:
                    path_aim.append([int(x), int(y), int(z)])
        data_vector = np.zeros((20, 20, 20))
        data_vector_connection=np.zeros((20,20,20))
        for sequantial in path_data:
            # print(sequantial[0]-1,sequantial[1]-1,sequantial[2]-1)
            data_vector[sequantial[0] - 1][sequantial[1] - 1][sequantial[2] - 1] = 1
            data_vector_connection[sequantial[0]-1][sequantial[1]-1][sequantial[2]-1]=sequantial[3]
        for sequantial in obstacle_data:
            if (sequantial[0]>20)or(sequantial[1]>20)or(sequantial[2]>20):
                continue
            data_vector[sequantial[0] - 1][sequantial[1] - 1][sequantial[2] - 1] = -1
        #out_array_1=numtest()
        out_array_1,out_array_2=judgement(data_vector_connection,path_start,path_aim)
        out_array_1=np.array(out_array_1)
        out_array_1=data_deal_std(out_array_1)
        out_array_2 = np.array(out_array_2)
        out_array_2 = data_deal_std(out_array_2)
        out_array_3,out_array_4=connection_judgement(data_vector_connection,path_start,path_aim)
        out_array_3=np.array(out_array_3)
        out_array_4=np.array(out_array_4)
        out_array_3=data_deal_std(out_array_3)
        out_array_4=data_deal_std(out_array_4)
        out_array_1=out_array_1.reshape(-1)
        out_array_2 = out_array_2.reshape(-1)
        out_array_3 = out_array_3.reshape(-1)
        out_array_4= out_array_4.reshape(-1)
        print(out_array_4.shape,out_array_3.shape, out_array_2.shape, out_array_1.shape)
        #print(out_array_1)
        #print(out_array_2)
        #print(out_array)
        out_array=np.concatenate((out_array_1, out_array_2,out_array_3,out_array_4), axis=0)
        #print(out_array.shape)
        path_data=[]
        n+=1
        save_path = 'other_data_feature27/scenedata{}.npy'.format(n)
        np.save(save_path, out_array)
        print(save_path)
score_path = 'scorenew23.npy'
np.save(score_path, score)
final_time=time.time()-start_time
print(final_time)
