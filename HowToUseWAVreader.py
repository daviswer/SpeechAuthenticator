import numpy as np
from WAVreader import *
from pylab import *
from sklearn.decomposition import PCA

filelist = ['6-1.wav',
            '3-1.wav'
            ]
data = WAVreader(filelist, [], 0.1)

X = [dat[0] for dat in data.concatset]
labels = [(dat[1]+0.0)/len(data.dataset) for dat in data.concatset]
pca = PCA(n_components=2)
pca.fit(X)
newX = pca.transform(X)
scatter([x[0] for x in newX], [x[1] for x in newX],c=labels,alpha=0.5)
