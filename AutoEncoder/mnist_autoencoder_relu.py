# encoding: utf-8  
#                            自动编码机 MNIST
#####################################################################
#from tensorflow.examples.tutorials.mnist 
import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)

#tensorflow图
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate=0.01
training_epochs=10
batch_size=256
display_step=1 #

n_input=784
n_h1=512  #隐藏层
n_h2=256
n_h3=128
#===========================================================================
#输入、输出 占位
#先指明输入数据的格式
X=tf.placeholder(tf.float32,[None,n_input])
#===========================================================================
#初始化函数 weight bias
def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
#===========================================================================
#===========================================================================
#                             autoencoder共三层
#--------------------------------------------
# initial weight and bias 
encoder_w1=weight_variable([n_input,n_h1])
encoder_w2=weight_variable([n_h1,n_h2])
encoder_w3=weight_variable([n_h2,n_h3])
decoder_w1=weight_variable([n_h3,n_h2])
decoder_w2=weight_variable([n_h2,n_h1])
decoder_w3=weight_variable([n_h1,n_input])

encoder_b1=bias_variable([n_h1])
encoder_b2=bias_variable([n_h2])
encoder_b3=bias_variable([n_h3])
decoder_b1=bias_variable([n_h2])
decoder_b2=bias_variable([n_h1])
decoder_b3=bias_variable([n_input])
#--------------------------------------------
# encoder layers 
def encoder(x):
    layer_1=tf.nn.relu(tf.add(tf.matmul(x,encoder_w1),encoder_b1))
    layer_2=tf.nn.relu(tf.add(tf.matmul(layer_1,encoder_w2),encoder_b2))
    layer_3=tf.nn.relu(tf.add(tf.matmul(layer_2,encoder_w3),encoder_b3))
    return layer_3
# decoder layers 
def decoder(x):
    layer_1=tf.nn.relu(tf.add(tf.matmul(x,decoder_w1),decoder_b1))
    layer_2=tf.nn.relu(tf.add(tf.matmul(layer_1,decoder_w2),decoder_b2))
    layer_3=tf.nn.relu(tf.add(tf.matmul(layer_2,decoder_w3),decoder_b3))
    return layer_3
# layer_1=tf.nn.relu(tf.matmul(x,encoder_w1)+encoder_b1)
#===========================================================================
#===========================================================================
# model
X_encoder=encoder(X)
X_decoder=decoder(X_encoder)
#===========================================================================
#训练和评估模型
cost=tf.reduce_mean(tf.pow(X_decoder-X,2)) #损失函数 均方误差
train_step=tf.train.RMSPropOptimizer(learning_rate).minimize(cost) #训练
#train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#train_step=tf.train.AdamOptimizer(learning_rate).minimize(cost)
#===========================================================================
#初始化变量
init=tf.global_variables_initializer()
#Session
sess=tf.Session()
sess.run(init)
#===========================================================================
#训练epochs次
total_batch=int(mnist.train.num_examples/batch_size)
for epoch in range(training_epochs):
    for i in range(total_batch):
        batch=mnist.train.next_batch(batch_size)
        _,c=sess.run([train_step,cost],feed_dict={X:batch[0]})
    if epoch%display_step==0:
        print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(c))
print("Optimization Finished!")

#===========================================================================
#测试集数据
X_encoder_decode=sess.run(X_decoder, feed_dict={X: mnist.test.images})
f,a=plt.subplots(10,10,figsize=(10,2))
for j in range(5):
    for i in range(10):
        a[j][i].imshow(np.reshape(mnist.test.images[j*10+i],(28,28)))
        a[j+5][i].imshow(np.reshape(X_encoder_decode[j*10+i],(28,28)))
plt.show()


