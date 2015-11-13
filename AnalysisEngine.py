
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import numpy as np
import cProfile
import random
from SNN import *
from NN import *
from WAVreader import *
from copy import deepcopy
from random import shuffle
from scipy.io import wavfile
from scipy.fftpack import fft, dct
from pylab import *
from numpy import dot, outer


# In[2]:

data = getMulticlassData('6-1.wav', 0, 5) + getMulticlassData('3-1.wav',3,5) + getMulticlassData('4-1.wav',4,5) + getMulticlassData('1-1.wav', 1, 5) + getMulticlassData('5-1.wav',2,5)
# data = getData('6-1.wav', -1) + getData('1-1.wav', 1)
print len(data)


# In[3]:

trainset = []
testset = []
for i in xrange(len(data)):
    if i%10 == 0: testset.append(data[i])
    else: trainset.append(data[i])
print (len(data), len(testset))


# In[ ]:




# In[ ]:

# NTN = ShallowNeuralNetwork(hidden_dim=27)#input_dim=26, hidden_dim=13)
# cProfile.run('NTN.train(trainset, maxiter=61, alpha=0.0001, display_progress=True)')


# In[5]:

# print NTN.accuracy(trainset), NTN.accuracy(testset)


# In[ ]:




# In[4]:

class SoftmaxNeuralNetwork:
    def __init__(self, input_dim=729, hidden_dim=27, output_dim=3, afunc=np.tanh, d_afunc=(lambda z : 1.0 - z**2)):        
        self.afunc = afunc 
        self.d_afunc = d_afunc      
        self.input = np.ones(input_dim)                                         
        self.hidden = np.ones(hidden_dim+1)      
        self.output = np.ones(output_dim)
        self.predictions = np.ones(output_dim)
        self.iweights = np.random.normal(scale=0.0001, size=(input_dim, hidden_dim))
        self.oweights = np.random.normal(scale=0.0001, size=(hidden_dim+1, output_dim)) 
        self.ierr = np.zeros(self.iweights.shape)
        self.oerr = np.zeros(self.oweights.shape)
        self.imom = np.zeros(self.iweights.shape)
        self.omom = np.zeros(self.oweights.shape)
        
    def forward_propagation(self, ex):
        self.input = ex
        self.hidden[:-1] = self.afunc(dot(self.input, self.iweights))
        self.output = np.exp(dot(self.hidden, self.oweights))
        self.output /= np.sum(self.output)
        self.output -= np.max(self.output)
        self.predictions = np.sign(self.output)+1
        
    def backward_propagation(self, labels, alpha=0.5):
        labels = np.array(labels)
        oerr = labels-self.predictions
        herr = dot(oerr, self.oweights.T) * self.d_afunc(self.hidden)
        self.oweights += alpha * outer(self.hidden, oerr)
        self.iweights += alpha * outer(self.input, herr[:-1])
        return 0.5 * np.sum(np.abs(labels-self.predictions))

    def train(self, training_data, maxiter=5000, alpha=0.05, lmbda=0, epsilon=1.5e-8, display_progress=False):       
        iteration = 0
        error = sys.float_info.max
        while error > epsilon and iteration < maxiter:
            gamma = 1/(2+math.trunc(np.sqrt(iteration)))
            error = 0.0
            size = 0.0
            shuffle(training_data)
            for ex, labels in training_data:
                self.forward_propagation(ex)
                size += abs(self.output)
                error += self.backward_propagation(labels, alpha=alpha)
                self.imom = self.ierr + gamma*self.imom
                self.omom = self.oerr + gamma*self.omom
#                 self.iweights += self.imom #- lmbda*self.l2penalty(self.iweights)
#                 self.oweights += self.omom #- lmbda*self.l2penalty(self.oweights)
                self.iweights = self.iweights*(1-lmbda)
                self.oweights = self.oweights*(1-lmbda)
            if display_progress and iteration%10==0:
                print 'completed iteration %s; error is %s; size is %s' % (iteration, error, np.sum(np.absolute(self.iweights)))
            iteration += 1
            
    def accuracy(self, data):
        score = 0.0
        for ex, label in data:
            self.forward_propagation(ex)
#             print self.predictions, np.array(label), self.predictions-np.array(label)
            if np.sum(np.multiply(self.predictions, np.array(label)))==1: score+=1
        return score/len(data)
            
    def l2penalty(self, arr):
        return np.multiply(np.array(map(lambda x: linalg.norm(arr[:,x]), range(len(arr[0])))), arr)
                    
    def predict(self, ex):
        self.forward_propagation(ex)
        return deepcopy(self.output)
        
    def hidden_representation(self, ex):
        self.forward_propagation(ex)
        return self.hidden


# In[5]:

SNN = SoftmaxNeuralNetwork(output_dim=5)
# print data[0][1]
# SNN.forward_propagation(data[0][0])
# SNN.backward_propagation(data[0][1])
cProfile.run('SNN.train(trainset, maxiter=401, alpha=0.002, lmbda=.0001, display_progress=True)')


# In[18]:

print SNN.accuracy(trainset), SNN.accuracy(testset)


# In[18]:

print SNN.accuracy(trainset), SNN.accuracy(testset)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

#SIMPLE TIMER
cProfile.run("rate, x = wavfile.read('5-1.wav');")
print rate, len(x)


# In[28]:

#MORE COMPLEX TIMER - BUT POSSIBLY CAN EXTRACT INFO INTO VARIABLES
import cProfile, pstats, StringIO
pr = cProfile.Profile()
pr.enable()
pr.run("rate, x = wavfile.read('5-1.wav');")
pr.disable()
s = StringIO.StringIO()
ps = pstats.Stats(pr, stream=s).strip_dirs()
ps.print_stats()
print s.getvalue()

