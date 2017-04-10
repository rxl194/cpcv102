# python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


# In[2]:

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset  = save['test_dataset']
    test_labels  = save['test_labels']
    del save
    print ("Train set: ", train_dataset.shape, train_labels.shape)
    print ("Validateion set: ", valid_dataset.shape, valid_labels.shape)
    print ("Test set: ", test_dataset.shape, test_labels.shape)    


# In[3]:

image_size = 28
num_labels = 10

def reformat(x, y):
    x = x.reshape([-1, image_size, image_size]).astype(np.float)
    y = (np.arange(num_labels) == y[:,None]).astype(np.float)
    return x, y

train_x, train_y = reformat(train_dataset, train_labels)
valid_x, valid_y = reformat(valid_dataset, valid_labels)
test_x,  test_y  = reformat(test_dataset,  test_labels)
print ("Train set: ", train_x.shape, train_y.shape)
print ("Validateion set: ", valid_x.shape, valid_y.shape)
print ("Test set: ", test_x.shape, test_y.shape) 


# In[4]:

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# In[5]:

def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)


# In[6]:

def conv2d(x, W):
    return tf.nn.conv2d(x, W, 
        strides=[1, 1, 1, 1], padding='SAME')


# In[7]:

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
        strides=[1, 2, 2, 1], padding='SAME')


# In[8]:

graph = tf.Graph()
with graph.as_default():        
    
    keep_prob = tf.placeholder( tf.float32 )
    tf_train_x = tf.placeholder( tf.float32, shape=[None, image_size, image_size])
    tf_train_y = tf.placeholder( tf.float32, shape=[None, num_labels])
    tf_train_image = tf.reshape( tf_train_x, [-1, image_size, image_size, 1])
    
    w1 = weight_variable([5, 5, 1, 32])
    b1 = bias_variable([32])
    h1_relu = tf.nn.relu( conv2d(tf_train_image, w1) + b1)
    h1 = max_pool_2x2( h1_relu )
    
    w2 = weight_variable([5, 5, 32, 64])
    b2 = bias_variable([64])
    h2_relu = tf.nn.relu( conv2d(h1, w2) + b2)
    h2 = max_pool_2x2( h2_relu )
    
    flat_size = int((image_size/4)*(image_size/4)*64)
    h2_flat = tf.reshape(h2, [-1, flat_size])
    
    w1_fc = weight_variable([flat_size, 1024])
    b1_fc = bias_variable([1024])
    h1_fc = tf.nn.dropout(
        tf.nn.relu( tf.matmul(h2_flat, w1_fc) + b1_fc ), keep_prob )
    
    w2_fc = weight_variable([1024, 10])
    b2_fc = bias_variable([10])    
    
    logits = tf.matmul(h1_fc, w2_fc) + b2_fc
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_y, logits=logits))
    
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    correct_pred = tf.equal( tf.argmax(logits, 1), tf.argmax(tf_train_y, 1) )
    accuracy = tf.reduce_mean ( tf.cast(correct_pred, tf.float32) )            


# In[ ]:

train_size = train_dataset.shape[0]
num_steps = 5001
batch_size = 100
num_batch = int(train_size/batch_size)
offset = 0

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print ("Initialized")
    for i in range(num_steps):
        batch_x = train_x[offset:offset+batch_size,:]
        batch_y = train_y[offset:offset+batch_size,:]
        offset += batch_size
        if ( offset >= train_size ): offset = 0
        if i % 100 == 0:
            train_accuracy = accuracy.eval( feed_dict = {
                tf_train_x: batch_x, tf_train_y: batch_y, keep_prob: 1.0 })
            print ('step %d, train accuracy %g ' % 
                   (i, train_accuracy))               
#            valid_accuracy = accuracy.eval( feed_dict = {
#                tf_train_x: valid_x, tf_train_y: valid_y, keep_prob: 1.0 })            
#            print ('step %d, train accuracy %g, valid accuracy %g' % 
#                   (i, train_accuracy, valid_accuracy))       
        optimizer.run( feed_dict = {
            tf_train_x: batch_x, tf_train_y: batch_y, keep_prob: 0.5 })
    print ('valid accuracy %g' % accuracy.eval(feed_dict = {
        tf_train_x: valid_x, tf_train_y: valid_y, keep_prob: 1.0}))    
    print ('test accuracy %g' % accuracy.eval(feed_dict = {
        tf_train_x: test_x, tf_train_y: test_y, keep_prob: 1.0}))


# In[ ]:



