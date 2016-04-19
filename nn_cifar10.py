#Example file of training a convolutional neural network
#to recognize image categories

import tensorflow as tf
import numpy as np
import cPickle
import matplotlib.pyplot as plt

#function to get cifar10 data
#returns a dictionary with key data and key labels
def unpickle(file):
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def get_batch(xdata, ydata, nbatch):
	N = len(ydata)
	inds = np.random.choice(N, size=nbatch, replace=False)
	xret = xdata[inds,:]
	labs = [ydata[i] for i in inds]
	yret = np.zeros((nbatch,10))
	yret = np.zeros((nbatch,10))
	for i in range(0,nbatch):
		yret[i,labs[i]] = 1
	return (xret,yret)

Nbatch = 50
Npix = 32
Nchannels = 3
Nlabels = 10
Nfilters = 32
std_init = 0.01
weight_decay = 0.000
lr = 1e-2
momentum = 0.9
#We will feed the data to the network at run time via a function
#so for now we need to create a tensorflow placeholder variable
#to be able to crate the NN layers

x_batch = tf.placeholder(np.float32, shape=(Nbatch, Npix, Npix, Nchannels), name="data")
y_batch = tf.placeholder(np.float32, shape=(Nbatch, Nlabels), name="labels")

#Now need to create a filter tensor to pass to a convolution layer
W1 = tf.Variable(tf.random_normal([3,3,3,Nfilters], stddev=std_init), name='W1')
b1 = tf.Variable(tf.zeros([Nfilters]), name='b1')

conv1 = tf.nn.conv2d(x_batch, W1, [1,1,1,1], padding="SAME")
conv1_bias = tf.nn.bias_add(conv1,b1, name='conv1bias')
out1 = tf.nn.relu(conv1_bias, name='out1')

#Now need to reshape the output of the conv layer to be able
#to feed it to a fully connected layer
in2 = tf.reshape(out1, [Nbatch, -1])
W2 = tf.Variable(tf.random_normal([Nfilters*Npix*Npix,Nlabels], stddev=std_init), name='W2')
b2 = tf.Variable(tf.zeros([Nlabels]), name='b2')
out2 = tf.matmul(in2,W2)+b2

#Now set up the loss function and regularization terms
#note that here we again use the y placeholder variable
class_loss = 1.0/Nlabels*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out2,y_batch, name='smaxloss'))

reg_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)

loss = class_loss + weight_decay*reg_loss

#Finally set up a optimizer to minimize the loss function
opt = tf.train.MomentumOptimizer(lr, momentum, name='mom')
train = opt.minimize(loss)

#now run the graph
init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)
data = unpickle("./data/cifar10/cifar-10-batches-py/data_batch_1")
N,W = data['data'].shape
mu = np.mean(data['data'], axis=0)
sig = np.std(data['data'], axis=0)

imgdata = np.zeros((N, Npix, Npix, Nchannels))
for i in range(0,3):
	imgdata[:,:,:,i] = np.reshape(data['data'][:,i*1024:(i+1)*1024], (N,32,32)).astype(np.float32)
	imgdata[:,:,:,i] = (imgdata[:,:,:,i]-125)/125



for step in xrange(1001):
	
	tup = get_batch(imgdata, data['labels'], Nbatch)
	sess.run(train, feed_dict={x_batch: tup[0], y_batch:tup[1]})
	
	if step % 20 == 0:
		print(sess.run(loss,feed_dict={x_batch: tup[0], y_batch:tup[1]}))