# encoding: utf-8  
##
#                      mnist数字文本识别
#####################################################################
#加载下载的数据
#from tensorflow.examples.tutorials.mnist 
import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)

#tensorflow图
import tensorflow as tf

#先指明输入数据的格式
x=tf.placeholder(tf.float32,[None,784])
#先指明正确的结果（lable，正确值）
y_=tf.placeholder("float",[None,10])

#参数变量  weight，bias
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
#模型构建
y=tf.nn.softmax(tf.matmul(x,W)+b)

#定义损失函数

#交叉熵
cross_entropy=-tf.reduce_sum(y_*tf.log(y))

#训练
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化变量
init=tf.global_variables_initializer()

#Session
sess=tf.Session()
sess.run(init)

#训练1000次
for i in range(1000):
	batch_xs,batch_ys=mnist.train.next_batch(50)
	sess.run(train_step,feed_dict={x: batch_xs,y_: batch_ys})

#模型评估
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
#测试集数据
feed_dict={x: mnist.test.images,y_: mnist.test.labels}
print(sess.run(accuracy, feed_dict=feed_dict))


#第一次成功率0.9197





