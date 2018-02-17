import os 
import tensorflow as tf 
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import *#进度条
#inputNodes是输入层节点数784
#outputNodes是输出层节点数为汉字个数
def tfrecord_read(filename, picsCounts, outputNodes, height, width):
    inputNodes = height * width
    input_images = np.array([[0]*inputNodes for i in range(picsCounts)])  
    input_labels = np.array([[0]*outputNodes for i in range(picsCounts)]) 
    filename_queue = tf.train.string_input_producer([filename]) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw' : tf.FixedLenFeature([], tf.string),
                                   })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28])
    label = tf.cast(features['label'], tf.int64)
    with tf.Session() as sess: #开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        index = 0
        print(filename + '正在读取中，请稍后...')
        for i in tqdm(range(picsCounts)):
            example, l = sess.run([image,label])#在会话中取出image和label
            for h in range(0, height):  
                for w in range(0, width):
                    input_images[index][w+h*width] = example[h][w]
            input_labels[index][l] = 1
            #img=Image.fromarray(input_images[index].reshape(28,28))#将array转化成图片，验证是否正确
            #img.show()
            index = index + 1
        coord.request_stop()
        coord.join(threads)
    return input_images, input_labels
            
