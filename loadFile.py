import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import math
import numpy as np

def getFileNames(imgType):		#generate all filenames for a chosen type, with a specified count of images
	tempallFileNames =  os.listdir("./101_ObjectCategories/" + imgType)	#get all the filenames as a list
	allFileNames = list()
	for f in tempallFileNames:
		allFileNames.append("./101_ObjectCategories/" + imgType + "/" + f)		
	
	numImages = len(allFileNames) 		#get the number of images found
	imgCount = int(round((0.7 * numImages),0))			#get 70% of the images (rounded)
	return allFileNames[:imgCount], allFileNames[imgCount:]
	
def getImageData(fileNameList,label):		#decode all the images from a given list of filenames
	outputData = list()
	outputLabels = list()
	for file in fileNameList:
		imageContent = tf.read_file(file)
		imageData = tf.image.resize_images(tf.image.decode_jpeg(imageContent,3),[56,56])
		outputData.append(tf.reshape(imageData,[-1]).eval())
		outputLabels.append(label)
	return np.array(outputData), np.array(outputLabels)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)	
	
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
planeFileNames, planeTestNames = getFileNames("airplanes")
bikeFileNames, bikeTestNames = getFileNames("motorbikes")
faceFileNames, faceTestNames = getFileNames("faces")

x = tf.placeholder(tf.float32,[None,9408], name="x")

W_conv1 = weight_variable([5,5,3,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,56,56,3], name="reshape_x")

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([12544, 12544])
b_fc1 = bias_variable([12544])

h_pool2_flat = tf.reshape(h_pool2, [-1, 12544], name="reshape_h_pool_flat")
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([12544,3])
b_fc2 = bias_variable([3])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#y_ = tf.placeholder(tf.float32, [None,3], name="y_")			#answers

#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = 	tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess,"./IMGModel/model")
	testImage, _ = getImageData(["./flycar.jpg"],[1,0,0])
	print(sess.run(y_conv, feed_dict={x:testImage, keep_prob:1.0}))
	answer = (sess.run(tf.nn.softmax(y_conv), feed_dict={x:testImage, keep_prob:1.0}))
	resultText = "NO_ANSWER"
	if(np.where(answer==1)[1] == 0):	
		resultText = "It's a plane!"
	elif(np.where(answer==1)[1] == 1):
		resultText = "It's a motorbike!"
	elif(np.where(answer==1)[1] == 2):
		resultText = "It's a face!"
	
	print(resultText)