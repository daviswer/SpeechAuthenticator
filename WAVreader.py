import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, dct
from pylab import *
from numpy import dot


class WAVreader:
    # Takes list of file names, list of labels to apply, and threshold as a percent of max volume for low-volume filtering. 
    # Labels should start at 0 and increase as new speakers are added. To use range(0,#ofFiles) as the labelling just input an empty list
    def __init__(self, filenames, labels, threshold):
        self.filenames = filenames
        self.labels = labels
        self.numClasses = 0
        self.length = 0
        self.FFTset = [] #The set of labeled Fourier Transforms
        self.MFset = [] #The set of labeled Mel Frequencies
        self.MFCCset = [] #The set of labeled MFCCs (with cosine transform)
        self.deltaset = [] #The set of labeled MFCC deltas
        self.lnMFCCset = [] #The set of labeled log-scaled MFCCs
        self.lnDeltaset = [] #The set of labeled log-scaled deltas
        self.concatset = [] #The set of labeled concatenations of log-scaled MFCCs and deltas, PLUS TENSOR DIAGONAL
        self.dataset = [] #concatset, but labeled in such a way as to allow softmax learning
        self.tconcatset = [] #The set of labeled concatenations of log-scaled MFCCs and deltas, PLUS THE FULL TENSOR
        self.tdataset = [] #concatset, but labeled in such a way as to allow softmax learning
        self.svmset = [] #The set of labeled concatenations, no tensor info, for use in SVMs
#        self.svmdeltaset = []
 #       self.svmfftset = []
  #      self.svmtestset = []
        
        #Get label info
        if len(labels)==0:
            self.labels = range(len(filenames))
        self.numClasses = max(self.labels)+1
        
        #MFCC calculation
        for key, filename in enumerate(filenames):
            
            #Read the file, split into windows
            rate, x = wavfile.read(filename);
            samplelen = rate*0.03
            samples = self.purge([x[i*samplelen/2:i*samplelen/2+samplelen]/100.0 for i in range(int(len(x)/samplelen*2))], threshold)
            
            #Fourier Transform
            ffts = [c[:samplelen/2-1] for c in map(lambda i: abs(fft(i)), samples)]
            
            #Get frequency range
            T = samplelen/rate
            frqLabel = map(lambda i: i/T, range(int(samplelen/2-1)))
            
            #Transform frequencies to Mel Frequencies
            melLabel = map(lambda i: 1125*log(1+i/700), frqLabel)
            melPoints = [(melLabel[-1]-1.0)/27*i for i in range(28)]
            melFilters = [[max(0, min(1.0/(melPoints[i+1]-melPoints[i])*(x-melPoints[i+1]+.0)+1.0,
                                    -1.0/(melPoints[i+2]-melPoints[i+1])*(x-melPoints[i+1]+.0)+1.0))
                         for x in melLabel] for i in range(26)]
            
            #Apply filters
            Mfs = [[np.dot(f, melFilters[i]) for i in range(26)] for f in ffts]
            
            #Apply cosine transform
            MFCCs = [dct(Mf) for Mf in Mfs]
            
            #Calculate deltas
            deltas = [[(MFCCs[i+1][j]+2*MFCCs[i+2][j]-MFCCs[i-1][j]-2*MFCCs[i-2][j])/10 
                      for j in range(26)] for i in range(2,len(MFCCs)-2)]
            
            #Perform log-scaling
            lnMFCCs = [[np.sign(c[i])*log(abs(c[i])/1000) for i in range(26)] for c in MFCCs]
            lnDeltas = [[np.sign(c[i])*log(abs(c[i])/10) for i in range(26)] for c in deltas]
            
            #Add data structures to cumulative, labeled data structures
            self.FFTset += [(ft, self.labels[key]) for ft in ffts]
            self.MFset += [(mf, self.labels[key]) for mf in Mfs]
            self.MFCCset += [(mfcc, self.labels[key]) for mfcc in MFCCs]
            self.deltaset += [(d, self.labels[key]) for d in deltas]
            self.lnMFCCset += [(mfcc, self.labels[key]) for mfcc in lnMFCCs]
            self.lnDeltaset += [(d, self.labels[key]) for d in lnDeltas]
            self.tconcatset += [(np.reshape(np.outer(np.append(lnMFCCs[i+2], [1]),np.append(lnDeltas[i], [1])), 729), self.labels[key]) 
                               for i in range(len(lnDeltas))]
            self.tdataset += [(np.reshape(np.outer(np.append(lnMFCCs[i+2], [1]),np.append(lnDeltas[i], [1])), 729),
                              [1 if j==self.labels[key] else 0 for j in range(self.numClasses)]) for i in range(len(lnDeltas))]
            self.concatset += [(np.concatenate((lnMFCCs[i+2],
                                           lnDeltas[i],
                                           np.outer(np.append(lnMFCCs[i+2], [1]),np.append(lnDeltas[i], [1])).diagonal())),
                            self.labels[key]) for i in range(len(lnDeltas))]
            self.dataset += [(np.concatenate((lnMFCCs[i+2],
                                            lnDeltas[i],
                                            np.outer(np.append(lnMFCCs[i+2], [1]),np.append(lnDeltas[i], [1])).diagonal())),
                             [1 if j==self.labels[key] else 0 for j in range(self.numClasses)]) for i in range(len(lnDeltas))]
            self.svmset += [(np.concatenate((lnMFCCs[i+2],
                                           lnDeltas[i],
                                           )),
                            self.labels[key]) for i in range(len(lnDeltas))]
#            self.svmtestset += [(np.concatenate((lnMFCCs[i],
 #                                          ffts[i],
  #                                         )),
   #                         self.labels[key]) for i in range(len(ffts))]
    #        self.svmdeltaset += [(d, self.labels[key]) for d in lnDeltas]
     #       self.svmfftset += [(f, self.labels[key]) for f in ffts]
            print "Finished file "+filename
        self.length = len(self.dataset)
        print
        print "Final data set consists of %d windows over %d classes" % (self.length, self.numClasses)


    def purge(self, dat, threshold):
        result = []
        t = threshold*np.max(map(np.max, dat))
        print t
        for window in dat:
            if max(np.abs(window)) > t: result.append(window)
        print "Scaled %d windows down to %d" % (len(dat), len(result))
        return result

    
