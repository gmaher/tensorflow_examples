#In this file we simulate some polynomial data with noise
#and then fit a second order polynomial to the data

import tensorflow as tf
import numpy as np

#create the data and add gaussian noise to the output Y
#note need the float 32 because numpy default is float64
#and tensorflow default is float32
X1 = np.random.rand(1,100).astype(np.float32)
X2 = X1**2
X = np.vstack((X1,X2))
Y = 2*X[0,:] + 0.2*X[1,:] + 1 + 0.05*np.random.normal(size=100)

#set up the tensorflow graph
W = tf.Variable(tf.random_uniform([2,1],0,4.0))
b = tf.Variable(tf.zeros([1]))


yhat = tf.matmul(X.T,W) + b

#set up loss function to minimize
loss = tf.reduce_mean(tf.square(Y-yhat)) + 0.45*b*b + 0.45*tf.reduce_mean(tf.square(W)) 
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#run the gradient descent by repeatedly calling sess.run(train)
#this runs the operation specified by the train variable above
for step in xrange(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(W), sess.run(b), sess.run(loss))

import matplotlib.pyplot as plt
plt.plot(X[1,:], Y, 'ro')
plt.plot(X[1,:], sess.run(yhat), 'gx')
plt.show()