# -*- coding:utf-8 -*-
'''
文件说明：训练好的模型使用，分为图搭建，训练好的模型导入，预测，预测结果生成文件几块
文件使用：直接python2调用'''
from inputData import *
import tensorflow as tf
import os
import numpy

TrainDataPath='TeamDataTrain.csv'
train=getdata(TrainDataPath)  #训练数据集
train_data=train[0]   #训练输入数据
train_label=train[1]  #训练实际应当输出结果
sess = tf.InteractiveSession()  #创建tf会话窗口

'''函数说明：获取初始权值，方差为0.1'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

'''函数说明：获取初始阈值，方差为0.1'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

'''函数说明：初始化卷积层，步长为1'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

'''函数说明：初始化池化层，窗口为2×2,步长为1'''
def max_pool_1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1,2, 2, 1], padding='SAME')


input_xlen=train_data.shape[1]  #输入参数个数
output_ylen=train_label.shape[1] #输出参数个数

x = tf.placeholder(tf.float32, [None, input_xlen]) #输入占位符
y_ = tf.placeholder(tf.float32, [None, output_ylen])#输出占位符
x_image = tf.reshape(x, [-1, 6, 10, 1]) #整理输入参数为6×10的二维图参数
keep_prob = tf.placeholder(tf.float32)  #dropout概率


'''卷积层，使用tanh激活'''
W_conv1 = weight_variable([3, 3, 1, 64])  # 3*3，1 channel，64 featuremap
b_conv1 = bias_variable([64])  # 64 featuremap bia
h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)  # tanh
h_pool1 = max_pool_1(h_conv1)  # pooling

'''隐藏层，128个隐藏神经元,使用tanh激活,dropout防止过拟合'''
W_fc1 = weight_variable([3*5*64, 128]) #128Neuros
b_fc1 = bias_variable([128])   #128Neuros bias
h_pool2 = tf.reshape(h_pool1, [-1,3*5* 64])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''输出层'''
W_fc2 = weight_variable([128, 2]) #2 output
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''计算损失量，优化'''
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-8,1.0)),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
tf.global_variables_initializer().run()

'''导入训练好的模型'''
ckpt=tf.train.get_checkpoint_state('./model/')
saver=tf.train.Saver()
saver.restore(sess,ckpt.model_checkpoint_path)

'''获取预测数据，并预测写入文件'''
TeamDataTestPath='TeamDataTest.csv'
pre_data=get_predata(TeamDataTestPath)
re=sess.run(y_conv,feed_dict={x:pre_data,keep_prob:1})
print "all data row{0}".format(len(re))
pre=[]
for i in range(len(re)):
    x=[]
    x.append(re[i][1])
    pre.append(x)
f=open('PredictPro.csv','wb')
tital=['主场赢得比赛的置信度']
writer=csv.writer(f)
writer.writerow(tital)
writer.writerows(pre)
print "all row{0}".format(len(pre))
f.close()