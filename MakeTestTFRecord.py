# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:00:13 2018

@author: 67636
制作test 的 tfrecords数据集
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:04:39 2018

@author: 67636
制作tfrecord数据集
"""
import os 
import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import *#进度条
 
#trainFile = 'H:/HWDB1/data/train/'
testFile = 'H:/MNIST/data/test/'
classes = [(file) for file in os.listdir(testFile)]
wordsCounts = len(classes)  #汉字总数
writer= tf.python_io.TFRecordWriter("nums_test.tfrecords") #要生成的文件
picsCounts = 0
for index,name in (enumerate(classes)):
    #print(name)
    class_path=testFile+name+'\\'
   # print(class_path)
    for img_name in os.listdir(class_path): 
        picsCounts = picsCounts + 1#图片总数
for index,name in (enumerate(classes)):
    #print(name)
    class_path=testFile+name+'\\'
   # print(class_path)
    for img_name in os.listdir(class_path): 
        #print(img_name)
        img_path=class_path+img_name #每一个图片的地址
 
        img=Image.open(img_path)
        
        img_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(index)])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串
    print('已完成%d'%index)
 
writer.close()


