# -*- coding:utf-8 -*-
'''文件作用：测试，预测时的数据获取
文件使用：在CNN.py中调用'''
import csv
import numpy as np
import pandas as pd


'''函数说明：数据归一处理，将数据线性归一到-1到1之间'''
def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

def z_score(x):
    x_mean=np.mean(x)
    s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]

def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]


'''函数说明：获取训练数据
函数参数：测试数据路径，此处为’TeamDataTrain.csv‘'''
def getdata(getfile):
    data=pd.read_csv(getfile).values
    data2=pd.read_csv('TeamData_win.csv').values
    result=[]
    x=[]
    y=[]
    for i in range(len(data)):
        x.append(Normalization2(data[i]))
        y.append(data2[i])

    result.append(np.array(x))
    result.append(np.array(y))
    return result

'''函数说明：模板测试数据获取
函数参数：测试数据路径，此处为’TeamDataTrain.csv‘'''
def gettestdata(testfile):
    data = pd.read_csv(testfile).values
    data2 = pd.read_csv('TeamData_win.csv').values
    result = []
    x = []
    y=[]
    manage=len(getdata(testfile)[0])
    print manage
    for i in range(1000):
        index=np.random.choice(manage)
        x.append(Normalization2(data[index]))
        y.append(data2[index])
    result.append(np.array(x))
    result.append(np.array(y))
    return result

'''函数说明：预测数据获取
函数参数：预测数据路径，此处为’TeamDataTest.csv‘'''
def get_predata(getfile):
    data = pd.read_csv(getfile).values
    x = []
    for i in range(len(data)):
        x.append(Normalization2(data[i]))
    return np.array(x)
