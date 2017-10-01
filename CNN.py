# -*- coding:utf-8 -*-
'''文件说明：训练模型搭建，使用一层卷积层，一层池化，一层全连接隐藏神经元，一层全连接输出
        卷积层：使用3×3卷积窗口，一个输入通道，64个卷积核
        池化层:2×2池化窗口，步长为1,不改变深度
        隐藏层：128个隐藏神经元
        输出层：2个输出，分别代表客场，主场胜负
文件使用：直接python2 调用'''
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

'''函数说明：预测结果准确率
函数参数：预测结果，实际结果'''
def myaccuracy(x1,x2):
    right=0
    n=x2.shape[0]
    for i in range(n):
        if x1[i][0]>x1[i][1]:
            if x2[i][0]>x2[i][1]:
                right=right+1
        elif x1[i][0]<x1[i][1]:
            if x2[i][0]<x2[i][1]:
                right=right+1
    return right*1.0/x2.shape[0]

'''开始训练，每50场比赛为一个batch进行训练，每迭代1000次测试当前训练结果,迭代10w次'''
for i in range(100000):
    rand_index = np.random.choice(len(train_data), size=50)
    rand_x = train_data[rand_index]
    rand_y = train_label[rand_index]
    if i % 1000 == 0:
        a=sess.run(y_conv,feed_dict={x:rand_x,keep_prob:1})
        train_accuracy=myaccuracy(a,rand_y)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: rand_x, y_:rand_y, keep_prob: 0.5})

'''训练结束，测试当前训练结果'''
test=gettestdata(TrainDataPath)
test_x=test[0]
test_y=test[1]
a=sess.run(y_conv,feed_dict={x:test_x,keep_prob:1})
print a
train_accuracy=myaccuracy(a,test_y)
print("training accuracy{0}".format(train_accuracy) )


'''
#predict and write the data into the PredicPro.csv
pre_data=get_predata('TeamData_pre3.csv')
re=sess.run(y_conv,feed_dict={x:pre_data,keep_prob:1})
print "all data re{0}".format(len(re))
pre=[]
for i in range(len(re)):
    x=[]
    x.append(re[i][1])
    pre.append(x)
f=open('PredictPro4.csv','wb')
tital=['主场赢得比赛的置信度']
writer=csv.writer(f)
writer.writerow(tital)
writer.writerows(pre)
print "all row{0}".format(len(pre))
'''

'''保存当前模型'''
saver = tf.train.Saver()
model_dir = "model"
model_name = "CNN_1"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
saver.save(sess, os.path.join(model_dir, model_name))
print("保存模型成功！")
