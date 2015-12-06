
# coding: utf-8

# In[25]:

get_ipython().magic(u'matplotlib inline')

from sklearn import svm
from WAVreader import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[4]:

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
clf.fit(standardX,Y) 
testX = [clip[0][:-1] for clip in testset]
testmeanX = np.mean(X, axis=0)
testvarX = np.var(X, axis=0)
teststandardX = np.divide(X-meanX, varX)
print clf.score(teststandardX, Y)


# In[28]:

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

ytrue = Y
ypred = []
for clip in teststandardX:
    ypred.append(clf.predict(clip))
confusion=confusion_matrix(ytrue, ypred)
cm_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
print('Confusion matrix')
print(confusion)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[6]:

# Plotting SVM

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * margin
yy_up = yy - a * margin

# plot the line, the points, and the nearest vectors to the plane
plt.figure(fignum, figsize=(4, 3))
plt.clf()
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
			facecolors='none', zorder=10)
plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

plt.axis('tight')
x_min = -4.8
x_max = 4.2
y_min = -6
y_max = 6

XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.figure(fignum, figsize=(4, 3))
plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
fignum = fignum + 1

plt.show() 

