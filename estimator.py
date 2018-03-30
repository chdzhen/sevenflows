"""
Convolutional Neural Network Estimator for facial landmark detection.
"""

import numpy as np
import tensorflow as tf
import cv2


tf.logging.set_verbosity(tf.logging.INFO)
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

def eval_input_fn():
    """ function for eval """
    return _input_fn('eval.TFRecords',2,1,False)

def predict_input_fn():
    ''' function for predict '''
    return _input_fn('test.TFRecords',2,1,False)


def cnn_model_fn(features,mode):
    '''
    the mode function for the network
    '''
    inputs = tf.to_float(features['x'],name='input_to_float')
    
    #===== layer 1 =====
    #Convolutional layer
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=16,
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
        filters=32,
        kernel_size=[3,3],
        strides=(1,1),
        padding='valid',
        activation=tf.nn.relu)

    #Pooling layer
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=(2,2),
        padding='valid'
    )
    
    #Convolutional layer
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=32,
        kernel_size=[3,3],
        strides=(1,1),
        padding='valid',
        activation=tf.nn.relu)
    
    #Pooling layer
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2,2],
        strides=(2,2),
        padding='valid')

    #===== layer 3 =====
    flatten = tf.layers.flatten(inputs=pool3)

    #dense layer 1,a fully connected layer
    dense1 = tf.layers.dense(
        inputs=flatten,
        units= 512,
        activation=tf.nn.relu,
        use_bias=True)
    dense1 = tf.layers.dropout(dense1,0.5)

    dense2 = tf.layers.dense(
        inputs= dense1,
        units = 256,
        activation=tf.nn.relu,
        use_bias=True
    )
    dense2 = tf.layers.dropout(dense2,0.2)

    #dense layer2,also known as the output layer
    logits = tf.layers.dense(
        inputs=dense2,
        units=17,
        activation=None,
        use_bias=True,
        name='logits')

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    labels = tf.to_int32(features['label'],name='label_to_int32')
   
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    
#------------------------------------------ costom estimator------------------------------------#

estimator = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='./train')


for i in range(200):
    print(i)
    mode = tf.estimator.ModeKeys.TRAIN
    estimator.train(input_fn=train_input_fn,steps=1000)

    mode = tf.estimator.ModeKeys.EVAL
    evaluation = estimator.evaluate(input_fn=eval_input_fn)
    print(evaluation)

'''
# train mode
mode = tf.estimator.ModeKeys.TRAIN
estimator.train(input_fn=train_input_fn,steps=200000)

#eval mode
mode = tf.estimator.ModeKeys.EVAL
evaluation = estimator.evaluate(input_fn=eval_input_fn)
print(evaluation)
'''
#predict mode
predictions = estimator.predict(input_fn=predict_input_fn)
for pre_dict in predictions:
    probability = pre_dict['probabilities']
    label = pre_dict['classes']
    print("================")
    print("probability: ")
    print(probability)
    print("classes: ")
    print(label)

