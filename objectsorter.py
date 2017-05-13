import tensorflow as tf
import sys
import numpy

# number of classes is 3 (planes, faces and moterbikes)
numClass = 2

# simple model (set to True) or convolutional neural network (set to False)
simpleModel = True

# dimensions of image (pixels)
height = 100
width = 100

'''
Instruction TensorFlow how to read the data
'''

def getImage(filename):
    # convert filenames to a queue for an input pipeline.
    filenameQ = tf.train.string_input_producer([filename], num_epochs=None)

    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example
    key, fullExample = recordReader.read(filenameQ)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        }
    )

    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg', [image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # cast image into a single array, where each element corresponds to the greyscale
    # value of a single pixel.
    # the "1-.." part inverts the image, so that the background is black.
    image = tf.reshape(1 - tf.image.rgb_to_grayscale(image), [height * width])

    # re-define label as a "one-hot" vector
    # it will be [1,0,0] or [0,1,0] or [0,0,1]
    # This approch can easily be extended to more classes.
    label = tf.stack(tf.one_hot(label -1, numClass))

    return label,image


'''
associate the "label" and "image" objects with the corresponding features read from 
a single example in the training data file
'''
trainLabel, trainImage = getImage("data/train-00000-of-00001")

# and similarly for the validation data
testLabel, testImage = getImage("data/validation-00000-of-00001")

'''
associate the "label_batch" and "image_batch" objects with a randomly selected batch
of labels and images respectively
'''
# for the traning data
trainImageBatch, trainLabelBatch = tf.train.shuffle_batch(
    [trainImage, trainLabel],
    batch_size=100,
    capacity = 320,
    min_after_dequeue=300  # this need to be lower then capacity
)
# for the validation data
testImageBatch, testLabelBatch = tf.train.shuffle_batch(
    [testImage, testLabel],
    batch_size=50,
    capacity=75,
    min_after_dequeue=70    # this need to be lower then capacity
)

sess = tf.InteractiveSession()

# x is the input array, which will contain the data from an image
# this creates a placeholder for x, to be populated later
x = tf.placeholder(tf.float32, [None, width * height])
# similarly, we have a placeholder for true outputs (obtained from labels)
y_ = tf.placeholder(tf.float32, [None, numClass])

'''
We are now ready to define the model
'''

if simpleModel:
    # run simple model y=Wx+b given in TensorFlow "MNIST" tutorial

    print("Running simple Model y=Wx+b")

    # initialise weights and biases to zero
    # weight maps input to output so is of size: (number of pixels) * (Number of Classes)
    weight = tf.Variable(tf.zeros([width* height, numClass]))
    # b is vector which has a size corresponding to number of classes
    b = tf.Variable(tf.zeros([numClass]))

    # define output calc (for each class) y = softmax(Wx+b)
    #softmax gives probability distribution across all classes
    y = tf.nn.softmax(tf.matmul(x, weight) + b)

'''
Before we start training we need to define the error, train step,
correct prediction and accuracy 
'''

# measure of error of our model

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# cros_entropy  needs to be minimised by adjusting weight and b

# define training step which minimises cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax gives index of hightest entry in vector (1st axis of 1D tensor)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

# get mean of all entries in correct prediction, the higher the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
Initialise and run the Training
'''

# initialize the variables
sess.run(tf.global_variables_initializer())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# start training
numSteps= 1000
for i in range(numSteps):

    batch_Image_xs, batch_ys = sess.run([trainImageBatch, trainLabelBatch])

    # run the training step with feed of images
    if simpleModel:
        train_step.run(feed_dict={x: batch_Image_xs, y_: batch_ys})


    if (i+1)%100 == 0: # then perfor validation

        # get validation batch
        testBatch_xs, testBatch_ys = sess.run([testImageBatch, testLabelBatch])

        if simpleModel:
            train_accuracy = accuracy.eval(
                feed_dict={x:testBatch_xs, y_:testBatch_ys}
            )

        print("step %d, training accuracy %g"%(i+1, train_accuracy))

# finalise
coord.request_stop()
coord.join(threads)







