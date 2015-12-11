import numpy as np
import random
import sys
import math
from copy import deepcopy
from random import shuffle, sample
from numpy import dot, outer

class QLNetwork:
    def __init__(self, trainset = None, predictor = None, num_classes = 2, reward = [], gamma = 0.9, afunc=np.tanh, d_afunc=(lambda z : 1.0 - z**2)):
        self.trainset = trainset
        self.predictor = predictor
        self.afunc = afunc 
        self.d_afunc = d_afunc
        self.numClasses = num_classes
        
        #Markov Process structures
        self.preds = np.ones(num_classes)
        self.numSamples = 0
        self.goodReward, self.badReward, self.sampleReward = reward
        self.done = False
        self.curSpeaker = 0
        self.gamma = gamma
        
        #NN structures
        self.input = np.zeros(num_classes+3)
        self.hidden = np.ones(num_classes)
        self.output = np.ones(1)
        self.iweights = np.random.normal(scale=0.0001, size=(num_classes+3, num_classes))
        self.oweights = np.random.normal(scale=0.0001, size=(num_classes, 1)) 
        self.ierr = np.zeros(self.iweights.shape)
        self.oerr = np.zeros(self.oweights.shape)
        self.tracker = [0]
        self.errtracker = []
        self.timetracker = [0]
        
    def getQ(self, action):
        self.input = np.append(np.sort(self.preds), [self.numSamples, action,1])
        self.hidden = self.afunc(dot(self.input, self.iweights))
        self.output = self.afunc(dot(self.hidden, self.oweights))
        return self.output[0]
        
    def step(self, alpha=0.5, epsilon=0.5, super_verbose=False):
        #GET EPSILON GREEDY ACTION
        action = 0
        ran = False
        if epsilon < random.random():
            #INTENTIONAL
            action = -1 if self.getQ(-1)>self.getQ(1) else 1
        else:
            #RANDOM
            ran = True
            action = random.sample([-1,1],1)[0]
        #MAKE PREDICTION
        pred = self.getQ(action)
        reward = 0
        Qprime = 0
        #APPLY ACTION
        if action==1 or self.numSamples==len(self.trainset[self.curSpeaker]):
            #PREDICT
            choice = np.argmax(self.preds)
            if choice==self.curSpeaker and not np.array_equal(self.preds[0]*np.ones(self.numClasses), self.preds):
                #CORRECT
                reward = self.goodReward
                self.tracker.append(1)#self.tracker[-1]+1)
            else:
                #INCORRECT
                reward = self.badReward
                self.tracker.append(0)#self.tracker[-1]-1)
            self.done = True
            self.timetracker.append((self.timetracker[-1]+self.numSamples)/2)
        else:
            #RESAMPLE
            probs = self.predictor(self.trainset[self.curSpeaker][self.numSamples][0])
            self.numSamples += 1
            self.preds = self.preds + probs
            reward = self.sampleReward
            Qprime = self.getQ(0) if self.getQ(0)>self.getQ(1) else self.getQ(1)
        #GET ERROR, PERFORM BACKPROP
        oerr = reward+self.gamma*Qprime-pred
        if self.done and super_verbose:
            print "Outer error is "+str(oerr)+", confidences are "+str(sorted(self.preds))+", result is "+str(reward)+" "+str(ran)
        herr = dot(oerr, self.oweights.T) * self.d_afunc(self.hidden)
        self.oweights += alpha * outer(self.hidden, oerr)
        self.iweights += alpha * outer(self.input, herr)
        self.errtracker.append(np.sum(0.5*oerr**2))
    
    def predict(self, speakerarr):
        action = -1 if self.getQ(-1)>self.getQ(1) else 1
        if action==1 or self.numSamples==len(speakerarr):
            #CONCLUDE, RETURN TO CHECKER
            self.done = True
        else:
            #RESAMPLE
            probs = self.predictor(speakerarr[self.numSamples][0])
            self.numSamples += 1
            self.preds = self.preds + probs

    def train(self, maxruns=100, annealAlpha = lambda x: 0.001, annealEpsilon = lambda x: 0.5, epsilon=1.5e-8, 
              progress_interval=1000, super_verbose=False, lmbda = 0.1):       
        run = 0
        error = sys.float_info.max
        while error > epsilon and run < maxruns:
            for speaker in self.trainset:
                shuffle(speaker)
            alpha = annealAlpha(run)
            self.iweights *= 1-alpha*lmbda
            self.oweights *= 1-alpha*lmbda
            for i in range(self.numClasses):
                self.curSpeaker = i
                while(self.done == False):
                    self.step(alpha=alpha, epsilon=annealEpsilon(run), super_verbose = super_verbose)
                #RESET FOR NEW RUN
                self.numSamples = 0
                self.done = False
                self.preds = np.ones(self.numClasses)
                
            run += 1
            if run%progress_interval==0:
                print 'completed iteration %s; size is %s' % (run, np.sum(np.absolute(self.iweights)))
            
    def test(self, testset, maxiter=100):
        score = 0
        runtime = 0
        mintime = 1000000
        maxtime = -1
        for speaker in testset:
            for _ in xrange(maxiter):
                #RESET
                shuffle(speaker)
                self.done = False
                self.numSamples = 0
                self.preds = np.ones(self.numClasses)
                #RESAMPLE UNTIL DONE
                while self.done == False:
                    self.predict(speaker)
                #TEST PREDICTION
                choice = np.argmax(self.preds)
                runtime += self.numSamples
                mintime = min(mintime, self.numSamples)
                maxtime = max(maxtime, self.numSamples)
                score += 1 if np.argmax(self.preds)==speaker[0][1] else 0
        return ((score+0.0)/(maxiter*self.numClasses), (runtime+0.0)/(maxiter*self.numClasses),
               mintime, maxtime)
            
