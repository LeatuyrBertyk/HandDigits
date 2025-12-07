# Khai báo thư viện
import os
import numpy as np

def loadMnist(path,kind='train'):
	labelsPath=os.path.join(path,'%s-labels.idx1-ubyte' % kind)
	imagesPath=os.path.join(path,'%s-images.idx3-ubyte' % kind)

	with open(labelsPath,'rb') as lbpath:
		lbpath.read(8)
		buffer=lbpath.read()
		labels=np.frombuffer(buffer,dtype=np.uint8)
	with open(imagesPath,'rb') as imgpath:
		imgpath.read(16)
		buffer=imgpath.read()
		images=np.frombuffer(buffer,dtype=np.uint8).reshape(len(labels),28,28).astype(np.float64)
	return images,labels
