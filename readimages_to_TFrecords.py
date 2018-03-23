"""
read 17 kinds of images to TFrecords 
"""
import os 
import tensorflow as tf 
import numpy as np
import cv2
import random

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_image_path():
    '''
    将图片分成train、eval、test三个部分
    '''
    cwd = os.getcwd()
    classes = os.listdir(cwd+"/"+"images_17flowers")

    image_path=[]


    for index,name in enumerate(classes):
        class_path = cwd+"/images_17flowers/"+name+'/'
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path.append([class_path + image_name,int(name)])

    train_image_path = random.sample(image_path,int(0.8*len(image_path)))
    image_path = [i for i in image_path if i not in train_image_path]
    eval_image_path = random.sample(image_path,int(0.8*len(image_path)))
    test_image_path = [i for i in image_path if i not in eval_image_path]
    print(len(train_image_path))
    print(len(eval_image_path))
    print(len(test_image_path))
    #返回三个集合的路径list
    return train_image_path,eval_image_path,test_image_path


def writeTFRecords(filename,train_image_path):
    '''
    读取图片并存为TFRecord文件
    '''
    writer = tf.python_io.TFRecordWriter(filename)
    for example_test in train_image_path:
        image = cv2.imread(example_test[0])
        image = cv2.resize(image,(200,200),interpolation=cv2.INTER_LINEAR)
        #cv2.imshow('test',image)
        #cv2.waitKey()
        image_raw = image.tostring()
        width,heigth,channel = image.shape
        print(example_test)
        label = example_test[1]
        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_raw':_bytes_feature(image_raw),
                    'image_label':_int64_feature(label),
                    'image_w':_int64_feature(width),
                    'image_h':_int64_feature(heigth),
                    'image_c':_int64_feature(channel)
                }
            )
        )
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__=='__main__':
    train_image_path,eval_image_path,test_image_path = get_image_path()
    writeTFRecords('train.TFRecords',train_image_path)
    writeTFRecords('eval.TFRecords',eval_image_path)
    writeTFRecords('test.TFRecords',test_image_path)
