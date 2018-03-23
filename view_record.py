'''
show the image of TFrecord
'''
import numpy as np
import tensorflow as tf
import cv2

def _extract_feature(element):
    """
    Extract features from a single example from dataset.
    """
    # Defaults are not specified since both keys are required.
    keys_to_features = {
        'image_raw':tf.FixedLenFeature([], tf.string),
        'image_label':tf.FixedLenFeature([],tf.int64),
        'image_w':tf.FixedLenFeature([],tf.int64),
        'image_h':tf.FixedLenFeature([],tf.int64),
        'image_c':tf.FixedLenFeature([],tf.int64),
    }
    
    features = tf.parse_single_example(element, keys_to_features)
    image_raw = tf.decode_raw(features['image_raw'],tf.uint8)
    image_label = features['image_label']
    image_w = features['image_w']
    image_h = features['image_h']
    image_c = features['image_c']
    image_tensor = tf.reshape(image_raw,[200,200,3])
    return {'x':image_tensor,'label':image_label}

def show_record(filenames):
    """
    Show the TFRecord contents
    """
    # Generate dataset from TFRecord file.
    dataset = tf.data.TFRecordDataset(filenames)
    
    dataset = dataset.map(_extract_feature)

    dataset = dataset.batch(20)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(10)

    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    # Actrual session to run the graph.
    with tf.Session() as sess:
        while True:
            try:
                image,label = sess.run([features['x'],features['label']])
                print(len(image))
                print(type(image))
                print(label)
                print(image[0].shape)
                for i in range(20):#batchsize
                    print(label[i])
                    cv2.putText(image[i],str(label[i]),(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv2.imshow("image", image[i])
                    cv2.waitKey()
            except tf.errors.OutOfRangeError:
                break
               
if __name__=='__main__':
    show_record('train.TFRecords')
    show_record('eval.TFRecords')
    show_record('test.TFRecords')