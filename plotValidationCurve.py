import numpy as np
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import Ridge
from WAVreader import *
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm

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
info = WAVreader(filelist, [0,0,5,5,7,7,3,3,4,4,1,1,6,6,2,2,8,8], 0.05)
data = info.svmset

C= 1.0 #regularizaton parameter

trainset = []
testset = []

for i in xrange(len(data)):
    if i%10 == 2: testset.append(data[i])
    else: trainset.append(data[i])
print len(trainset),len(testset)



# In[27]:

clf = svm.SVC()
X = np.array([clip[0][:-1] for clip in trainset])
meanX = np.mean(X, axis=0)
varX = np.var(X, axis=0)
standardX = np.divide(X-meanX, varX)
Y = np.array([clip[1] for clip in trainset])

np.random.seed(0)
param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    SVC(), X, Y, param_name="gamma", param_range=param_range,cv=10, scoring="accuracy", n_jobs=1)
#train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, labels, cv=5)
#print train_sizes
print train_scores           
print valid_scores         

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.legend(loc="best")
plt.show()  
