import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, dct
from pylab import *
from numpy import dot


class WAVreader:
    # Takes list of file names, list of labels to apply, and minimum threshold for low-volume filtering. 
    # Labels should start at 0 and increase as new speakers are added. To use range(0,#ofFiles) as the labelling just input an empty list
    def __init__(self, filenames, labels, threshold):
        self.filenames = filenames
        self.labels = labels
        self.FFTset = [] #The set of labeled Fourier Transforms
        self.MFset = [] #The set of labeled Mel Frequencies
        self.MFCCset = [] #The set of labeled MFCCs (with cosine transform)
        self.deltaset = [] #The set of labeled MFCC deltas
        self.lnMFCCset = [] #The set of labeled log-scaled MFCCs
        self.lnDeltaset = [] #The set of labeled log-scaled deltas
        self.concatset = [] #The set of labeled concatenations of log-scaled MFCCs and deltas
        self.dataset = [] #concatset, but labeled in such a way as to allow softmax learning
        
        if len(labels)==0:
            self.labels = range(len(filenames))
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
            self.FFTset += [(ft, labels[key]) for ft in ffts]
            self.MFset += [(mf, labels[key]) for mf in Mfs]
            self.MFCCset += [(mfcc, labels[key]) for mfcc in MFCCs]
            self.deltaset += [(d, labels[key]) for d in deltas]
            self.lnMFCCset += [(mfcc, labels[key]) for mfcc in lnMFCCs]
            self.lnDeltaset += [(d, labels[key]) for d in lnDeltas]
            self.concatset += [(np.reshape(np.outer(np.append(lnMFCCs[i+2], [1]),np.append(lnDeltas[i], [1])), 729), labels[key]) 
                               for i in range(len(lnDeltas))]
            self.dataset += [(np.reshape(np.outer(np.append(lnMFCCs[i+2], [1]),np.append(lnDeltas[i], [1])), 729),
                              [1 if j==labels[key] else 0 for j in range(len(filenames))]) for i in range(len(lnDeltas))]
            print "Finished file "+filename
        print
        print "Final data set consists of %d windows over %d classes" % (len(self.dataset), max(self.labels))


    def purge(self, dat, threshold):
        result = []
        log = ""
        for window in dat:
            total = np.sum(window)
            if abs(total) > threshold: result.append(window)
            #log += str(total)+" "
        print "Scaled %d windows down to %d" % (len(dat), len(result))
        #print
        #print log
        return result

    
