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
    image_tensor = tf.reshape(image_raw,[36,36,3])
    image_label = tf.cast(image_label,tf.float32)
    return {'x':image_tensor,'label':image_label}

def _input_fn(filenames,batch_size,num_epochs=None, shuffle=True):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_extract_feature)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=10000)
    if batch_size !=1:
        dataset = dataset.batch(batch_size)
    if num_epochs !=1:
        dataset = dataset.repeat(num_epochs)
    # Make dataset iteratable.
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features

def train_input_fn():
    """
    function for training
    """
    return _input_fn('train.TFRecords',20,5,True)


features = train_input_fn()

'''
the mode function for the network
'''
inputs = tf.to_float(features['x'],name='input_to_float')
labels = tf.to_int32(features['label'],name='label_to_int32')
labels = tf.one_hot(labels-1, 17,on_value=1.0, off_value=0.0,axis=-1)
'''
haha = tf.argmax(labels1,1)
with tf.Session() as sess:
    labels,labels1,haha = sess.run([labels,labels1,haha])
    print(labels)
    print(labels1)
    print(haha)
'''



#===== layer 1 =====
#Convolutional layer
conv1 = tf.layers.conv2d(
    inputs=inputs,
    filters=32,
    kernel_size=[3,3],
    strides=(1,1),
    padding='valid',
    activation=tf.nn.relu)
#Pooling layer
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2,2],
    strides=(2,2),
    padding='valid')

#===== layer 2 =====
#Convolutional layer
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[3,3],
    strides=(1,1),
    padding='valid',
    activation=tf.nn.relu)
#Convolutional layer
conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=64,
    kernel_size=[3,3],
    strides=(1,1),
    padding='valid',
    activation=tf.nn.relu)

#Pooling layer
pool2 = tf.layers.max_pooling2d(
    inputs=conv3,
    pool_size=[2,2],
    strides=(2,2),
    padding='valid')

#===== layer 3 =====
flatten = tf.layers.flatten(inputs=conv3)

#dense layer 1,a fully connected layer
dense1 = tf.layers.dense(
    inputs=flatten,
    units= 1024,
    activation=tf.nn.relu,
    use_bias=True)

#dense layer2,also known as the output layer
logits = tf.layers.dense(
    inputs=dense1,
    units=17,
    activation=tf.nn.softmax,
    use_bias=True,
    name='logits')


# Compute prdictions
predict_classes = tf.argmax(logits,1)

# Make prediction for PREDICTION mode
predictions_dict = {
    'class_ids':predict_classes[:tf.newaxis],
    'probabilities':tf.nn.softmax(logits),
    'logits':logits}

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    predictions_dict,predict_classes = sess.run([predictions_dict,predict_classes])
    print(predictions_dict['class_ids'])
    print(predictions_dict['probabilities'])
    print(predict_classes)









# caculate loss using mean squared error
loss = tf.losses.softmax_cross_entropy(labels,logits)

# Compute prdictions
predict_classes = tf.argmax(logits,1)

correct_prediction = tf.equal(tf.arg_max(labels,1),predict_classes)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#accuracy = tf.metrics.accuracy(labels=tf.arg_max(labels,1),predictions=predict_classes,name='acc_op')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    loss,correct_prediction = sess.run([loss,correct_prediction])
    print(loss)
    print(correct_prediction)