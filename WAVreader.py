import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, dct
from pylab import *
from numpy import dot

def purge(dat):
    result = []
    for window in dat:
        if np.sum(window) > 2: result.append(window)
    return result

    
def getData(filename, label, filter=False):
    
    def plots(i):
        print "WAV DATA, FFT, MEL CEPSTRALS, SCALED CEPSTRALS, DELTAS, SCALED DELTAS"
        figure(0)
        plot(samples[i])
        figure(1)
        plot(melLabel, ffts[i])
        figure(2)
        plot(MFCCs[i])
        figure(3)
        plot(lnMFCCs[i])
        figure(4)
        plot(deltas[i-2])
        figure(5)
        plot(lnDeltas[i-2])
        show()
    
    print
    print "Reading file "+filename
    rate, x = wavfile.read(filename);
    samplelen = rate*0.03
    samples = [x[i*samplelen/2:i*samplelen/2+samplelen]/100.0 for i in range(int(len(x)/samplelen*2))]
    print "Sample split into %d windows of length %d" % (len(samples), samplelen)
    if filter==True: 
        samples = purge(samples)
        print "Filtered down to %d informative windows" % (len(samples))
    ffts = [c[:samplelen/2-1] for c in map(lambda i: abs(fft(i)), samples)]
    T = samplelen/rate
    frqLabel = map(lambda i: i/T, range(int(samplelen/2-1)))
    print "Frequencies range from %f to %f" % (frqLabel[0], frqLabel[-1])
    melLabel = map(lambda i: 1125*log(1+i/700), frqLabel)
    print "Mel frequencies range from %f to %f" % (melLabel[0], melLabel[-1])
    melPoints = [(melLabel[-1]-1.0)/27*i for i in range(28)]
    print
    print "Building filters..."
    melFilters = [[max(0, min(1.0/(melPoints[i+1]-melPoints[i])*(x-melPoints[i+1]+.0)+1.0,
                            -1.0/(melPoints[i+2]-melPoints[i+1])*(x-melPoints[i+1]+.0)+1.0))
                 for x in melLabel] for i in range(26)]
    print "Applying filters..."
    Mfs = [[np.dot(f, melFilters[i]) for i in range(26)] for f in ffts]
    print "Applying cosine transform..."
    MFCCs = [dct(Mf) for Mf in Mfs]
    print "Calculating deltas..."
    deltas = [[(MFCCs[i+1][j]+2*MFCCs[i+2][j]-MFCCs[i-1][j]-2*MFCCs[i-2][j])/10 
              for j in range(26)] for i in range(2,len(MFCCs)-2)]
    print "Scaling..."
    lnMFCCs = [[np.sign(c[i])*log(abs(c[i])/1000) for i in range(26)] for c in MFCCs]
    lnDeltas = [[np.sign(c[i])*log(abs(c[i])/10) for i in range(26)] for c in deltas]
    print "Merging..."
    result = [(np.reshape(np.outer(np.append(lnMFCCs[i+2], [1]),np.append(lnDeltas[i], [1])), 729),
               label) for i in range(len(lnDeltas))]
#     result = [(mfcc, label) for mfcc in lnMFCCs]
    print "Done!"
    return result


def getMulticlassData(filename, label, numcat, filter=False):
    
    print
    print "Reading file "+filename
    rate, x = wavfile.read(filename);
    samplelen = rate*0.03
    samples = [x[i*samplelen/2:i*samplelen/2+samplelen]/100.0 for i in range(int(len(x)/samplelen*2))]
    print "Sample split into %d windows of length %d" % (len(samples), samplelen)
    if filter==True: 
        samples = purge(samples)
        print "Filtered down to %d informative windows" % (len(samples))
    ffts = [c[:samplelen/2-1] for c in map(lambda i: abs(fft(i)), samples)]
    T = samplelen/rate
    frqLabel = map(lambda i: i/T, range(int(samplelen/2-1)))
    print "Frequencies range from %f to %f" % (frqLabel[0], frqLabel[-1])
    melLabel = map(lambda i: 1125*log(1+i/700), frqLabel)
    print "Mel frequencies range from %f to %f" % (melLabel[0], melLabel[-1])
    melPoints = [(melLabel[-1]-1.0)/27*i for i in range(28)]
    print
    print "Building filters..."
    melFilters = [[max(0, min(1.0/(melPoints[i+1]-melPoints[i])*(x-melPoints[i+1]+.0)+1.0,
                            -1.0/(melPoints[i+2]-melPoints[i+1])*(x-melPoints[i+1]+.0)+1.0))
                 for x in melLabel] for i in range(26)]
    print "Applying filters..."
    Mfs = [[np.dot(f, melFilters[i]) for i in range(26)] for f in ffts]
    print "Applying cosine transform..."
    MFCCs = [dct(Mf) for Mf in Mfs]
    print "Calculating deltas..."
    deltas = [[(MFCCs[i+1][j]+2*MFCCs[i+2][j]-MFCCs[i-1][j]-2*MFCCs[i-2][j])/10 
              for j in range(26)] for i in range(2,len(MFCCs)-2)]
    print "Scaling..."
    lnMFCCs = [[np.sign(c[i])*log(abs(c[i])/1000) for i in range(26)] for c in MFCCs]
    lnDeltas = [[np.sign(c[i])*log(abs(c[i])/10) for i in range(26)] for c in deltas]
    print "Merging..."
    result = [(np.reshape(np.outer(np.append(lnMFCCs[i+2], [1]),np.append(lnDeltas[i], [1])), 729),
               [1 if j==label else 0 for j in range(numcat)]) for i in range(len(lnDeltas))]
#     result = [(mfcc, label) for mfcc in lnMFCCs]
    print "Done!"
    return result