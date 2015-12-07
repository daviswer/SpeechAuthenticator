
# coding: utf-8

# In[29]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import cProfile
import random
from SNN_test import *
from NN import *
from WAVreader import *
from copy import deepcopy
from random import shuffle
from scipy.io import wavfile
from scipy.fftpack import fft, dct
from pylab import *
from numpy import dot, outer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


# In[30]:

filelist = ['Audio/1-1.wav',
            'Audio/1-2.wav',
            'Audio/2-1.wav',
            'Audio/2-2.wav',
            'Audio/3-1.wav',
            'Audio/3-2.wav',
            'Audio/4-1.wav',
            'Audio/4-2.wav',
            'Audio/5-1.wav',
            'Audio/5-2.wav',
            'Audio/6-1.wav',
            'Audio/6-2.wav',
            'Audio/7-1.wav',
            'Audio/7-2.wav',
            'Audio/8-1.wav',
            'Audio/8-2.wav',
            'Audio/9-1.wav',
            'Audio/9-2.wav',
            ]
info = WAVreader(filelist, [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8], 0.05)
data = info.dataset


# In[31]:

trainset = []
testset = []
for i in xrange(len(data)):
    if i%10 < 3: testset.append(data[i])
    else: trainset.append(data[i])
print (len(data), len(testset))


# In[ ]:




# In[40]:

NTN = ShallowNeuralNetwork(input_dim=79, hidden_dim=27)
NTN.train(trainset, maxiter=61, alpha=0.0001, lmbda=0.00001, display_progress=True)


# In[41]:

print NTN.accuracy(trainset), NTN.accuracy(testset)


# In[ ]:




# In[56]:

SNN = SoftmaxNeuralNetwork(input_dim=79, output_dim=info.numClasses)
# print data[0][1]
# SNN.forward_propagation(data[0][0])
# print SNN.predictions
# SNN.backward_propagation(data[0][1])
SNN.train(trainset, maxiter=151, display_progress=True, anneal=lambda x: .01/(100+50*x))


# In[60]:

# 2-class, alpha=.001, lmbda=.0001, t=101
print SNN.accuracy(trainset), SNN.accuracy(testset)


# In[61]:

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

ytrue = []
ypred = []
for clip in testset:
    SNN.forward_propagation(clip[0])
    ypred.append(np.argmax(SNN.predictions))
    ytrue.append(np.argmax(clip[1]))
confusion=confusion_matrix(ytrue, ypred)
cm_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
print('Confusion matrix')
print(confusion)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


# In[62]:

precisions = np.array([(confusion[i][i]+0.0)/np.sum(confusion, axis=0)[i] for i in range(info.numClasses)])
meanp = np.mean(precisions)
recalls = np.array([(confusion[i][i]+0.0)/sum(confusion[i]) for i in range(info.numClasses)])
meanr = np.mean(recalls)
F1 = 2*meanr*meanp/(meanr+meanp)
print meanp
print meanr
print F1


# In[ ]:




# In[ ]:

