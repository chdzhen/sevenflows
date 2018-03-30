tfrecords_file='train.TFRecords'

filename_queue = tf.train.string_input_producer([tfrecords_file])

reader = tf.TFRecordReader()

_,serialized_example = reader.read(filename_queue)

keys_to_features = {
        'image_raw':tf.FixedLenFeature([], tf.string),
        'image_label':tf.FixedLenFeature([],tf.int64),
        'image_w':tf.FixedLenFeature([],tf.int64),
        'image_h':tf.FixedLenFeature([],tf.int64),
        'image_c':tf.FixedLenFeature([],tf.int64),
    }
    
features = tf.parse_single_example(serialized_example, keys_to_features)
image = tf.decode_raw(features['image_raw'],tf.uint8)
image = tf.reshape(image,[200,200,3])

label = tf.cast(features['image_label'],tf.float32)
image_batch, label_batch = tf.train.batch([image,label],batch_size=2,num_threads=64,capacity=2000)

sess = tf.Session()

#切记一定要打开cooder，不然sess.run() 就会卡住。
coord = tf.train.Coordinator()  
threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

while True:
    try: 
        image_batch_temp = sess.run([image])
        
        cv2.imshow('test',image_batch_temp[0])
        cv2.waitKey(4)
        '''
        for i in range(2):
            cv2.imshow('test',image_batch_temp[i])
            cv2.waitKey(4)
        '''
    except tf.errors.OutOfRangeError:
        print('outrange')
coord.join(threads) 
sess.close()