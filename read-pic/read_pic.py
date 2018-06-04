# -*- coding: utf-8 -*-
import tensorflow as tf
import forward
import generate
import os
import numpy as np

num_examples = 13000
BATCH_SIZE = 100
LEARNING_RATE_BASE =  0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnsit_model"

def backward():

#
   # x = tf.placeholder(tf.float32,[BATCH_SIZE,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.NUM_CHANNELS])
   # y_ = tf.placeholder(tf.float32, [BATCH_SIZE,forward.IMAGE_SIZE,forward.IMAGE_SIZE,forward.NUM_CHANNELS])
    xs , ys =generate.get_tfrecord(BATCH_SIZE, isTrain=True)
    y = forward.forward(x,True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)	
    print('x: ',x)
    print('y_:',y_)
    print('y: ',y)


#
    #loss = tf.reduce_mean(tf.square(y-y_))
    ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=y_))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    print('2222: ',tf.square(y_-y))
    print('loss:',loss)
#
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss , global_step=global_step)
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.MomentumOptimizer(learning_rate,[0.1]).minimize(loss)
#
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
#
    saver = tf.train.Saver()
#
