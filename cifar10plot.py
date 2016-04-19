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
	print labs
	yret = np.zeros((nbatch,10))
	for i in range(0,nbatch):
		yret[i,labs[i]] = 1
	return (xret,yret)

data = unpickle("./data/cifar10/cifar-10-batches-py/data_batch_1")
N,W = data['data'].shape
mu = np.mean(data['data'], axis=0)
sig = np.std(data['data'], axis=0)

imgdata = np.zeros((N, 32, 32, 3))
for i in range(0,3):
	imgdata[:,:,:,i] = np.reshape(data['data'][:,i*1024:(i+1)*1024], (N,32,32)).astype(np.float32)
	imgdata[:,:,:,i] = (imgdata[:,:,:,i]-125)/125

img1 = data['data'][100,:]
img1 = np.reshape(img1, (3,32,32))
lab1 = data['labels'][0]

print 'label 1: ', lab1
plt.imshow(imgdata[200,:,:,:], vmin=-1,vmax=1)
plt.show()

tup = get_batch(imgdata, data['labels'], 12)
print tup[0].shape
print tup[1].shape
print tup[0]
print tup[1]