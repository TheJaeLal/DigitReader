import tensorflow as tf
import numpy as np
import mnist_loader
import random

#Weights Initializer
initializer = tf.contrib.layers.xavier_initializer()

batch_size = 100
alpha = 0.0005
epochs = 20

#Load Data
train_data,valid_data,test_data = mnist_loader.load_data_wrapper()

inputs = tf.placeholder(tf.float32,shape=[None,28,28,1])
Y = tf.placeholder(tf.float32,shape=[None,10])

#Filter_1
#-> filter_size = (5 x 5)
#-> input_channels or (channels_in_image) = 1 
#-> output_channels or (num_of_filters) = 8
w1_shape = [5,5,1,8]
w1 = tf.Variable(initializer(w1_shape))
#Biases (single value, thus shape is empty tensor [])
b1_shape = []
b1 = tf.Variable(initializer(b1_shape))


#Filter_2
#-> filter_size = (3 x 3)
#-> input_channels or (channels_in_prev_layer) = 8
#-> output_channels or (num_of_filters) = 16
w2_shape = [3,3,8,16]
w2 = tf.Variable(initializer(w2_shape))
#Biases (single value, thus shape is empty tensor [])
b2_shape = []
b2 = tf.Variable(initializer(b2_shape))

#1st Convolutional Layer
conv1 = tf.nn.relu(
                tf.add(
                    tf.nn.conv2d(input=inputs,filter=w1,padding='SAME',strides=[1,1,1,1]),
                    b1)
                )
#1st Pooling layer
pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#2nd Convolutional Layer
conv2 = tf.nn.relu(
                tf.add(
                    tf.nn.conv2d(input=pool1,filter=w2,padding='SAME',strides=[1,1,1,1]),
                    b2)
                )
#2nd Pooling Layer
pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#Reshape output from conv2 (batch_size,7,7,16) to (batch_size,784)
fc_inputs = tf.reshape(pool2,[-1,784])

#1st Fully Connceted layer input -> 784 neurons output -> 30 neurons
#Weights
w3_shape = [784,30]
w3 = tf.Variable(initial_value=initializer(w3_shape))
#Biases
b3_shape = [30]
b3 = tf.Variable(initial_value=initializer(b3_shape))
#Output Activation
a0 = tf.nn.relu(tf.add(tf.matmul(fc_inputs,w3),b3))


#2nd Fully Connceted layer input -> 30 neurons output -> 10 neurons
#Weights
w4_shape = [30,10]
w4 = tf.Variable(initial_value=initializer(w4_shape))
#Biases
b4_shape = [10]
b4 = tf.Variable(initial_value=initializer(b4_shape))
#Output Activation
a1 = tf.nn.softmax(tf.add(tf.matmul(a0,w4),b4))

#Calculate training loss
loss = tf.reduce_sum(tf.losses.log_loss(labels=Y,predictions=a1))

#Calculate testing accuracy
is_correct = tf.equal(tf.argmax(a1,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

#Minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
train = optimizer.minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for e in range(epochs):            
        
        loss_val = 0

        num_of_training_samples = train_data[0].shape[0]

        for b in range(0,num_of_training_samples,batch_size):
        
            training_X = train_data[0][b:b+batch_size].reshape(-1,28,28,1)
            training_Y = train_data[1][b:b+batch_size]
            
            feed_train = {inputs:training_X,Y:training_Y}
            
            sess.run(train,feed_dict=feed_train)
            
            loss_val += sess.run(loss,feed_dict=feed_train)

        #Test agains validation data to provide accuracy
        test_X = valid_data[0].reshape(-1,28,28,1)
        test_Y = valid_data[1]
        feed_test = {inputs:test_X,Y:test_Y}
        
        accuracy_val = sess.run(accuracy,feed_dict=feed_test)
        print("Epoch {} : train_loss = {:.5f} validation_accuracy = {:}".format(e,loss_val,accuracy_val*100))

    print("*****End of Training*****")

	#Testing against test data
    test_X = test_data[0].reshape(-1,28,28,1)
    test_Y = test_data[1]
    feed_test = {inputs:test_X,Y:test_Y}
    accurac_on_test = sess.run(accuracy,feed_dict=feed_test)
    print("Accuracy on test_data : {}".format(accurac_on_test*100))

