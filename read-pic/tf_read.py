# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import os


image_train_path = './photos/train/'
tfRecord_train = './data/picture_train.tfrecords'
image_test_path = './photos/test/'
tfRecord_test = './data/picture_test.tfrecords'
data_path = './data'
resize_height = 50
resize_width = 50
#»òÕßresize_length = 32


#定义函数，写入tfRecord子函数，待调用
def write_tfRecord(tfRecordName, image_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic =  0 #计数器
    for file in os.listdir(os.path.join(image_path,'input')): #提取文件夹内图片名称
        img = Image.open(os.path.join(image_path,'input',file)) #Image.open
        print(os.path.join(image_path,'input',file))
        label = Image.open(os.path.join(image_path,'output',file))
        img_raw = img.tobytes()#    
        label_raw = label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),'label_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw]))}))
        writer.write(example.SerializeToString()) #
        num_pic += 1
        print("the number of picture:{}".format(num_pic))
    writer.close() #
    print("write tfrecord successfully")

#
def generate_tfRecord():
    isExists = os.path.exists(data_path) #
    if not isExists: #
        os.makedirs(data_path) #
        print('The directory was created successfully')
    else: #
        print('directory already exists')
    write_tfRecord(tfRecord_train, image_train_path) #
    write_tfRecord(tfRecord_test, image_test_path) #

#
def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True) #
    reader = tf.TFRecordReader() #
    _, serialized_example = reader.read(filename_queue) #
    features = tf.parse_single_example(serialized_example, features={'label_raw':tf.FixedLenFeature([], tf.string),'img_raw':tf.FixedLenFeature([], tf.string)}) #
    img = tf.decode_raw(features['img_raw'], tf.uint8) #
    img = tf.reshape(img, [50,50,1])
    #img.set_shape([50,50,1])
    img = tf.cast(img, tf.float32) * (1. / 255) #
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label = tf.reshape(label, [50,50,1])
    label = tf.cast(label, tf.float32) * (1. / 255)
