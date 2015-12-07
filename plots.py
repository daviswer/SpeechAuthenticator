import numpy as np
import pylab 

#NTN
trainerr = [.965,.805,.628,.513,.375,.334,.199,.182]
testerr = [.804,.588,.466,.394,.301,.268,.161,.148]
improvement = [.608,.382,.288,.243,.161,.146,.041,.042]
F1 = [.804,.586,.461,.390,.325,.264,.197,.173]
classes = [2,3,4,5,6,7,8,9]

# SNN
train_accuracy = [ 
    0.828007518797, 
    0.576955903272 ,
    0.600795478334 ,
    0.458142576848 ,
    0.499664023653 ,
    0.431657754011 ,
    0.444187765663 , 
    0.362680045315 
    ]
test_accuracy = [
    0.812021857923, 
    0.565275016567, 
    0.589555880918, 
    0.451563691838, 
    0.467565026637, 
    0.406686626747, 
    0.408796895213,
    0.346168365421
    ]
f1_score = [
    0.818175880441, 
    0.627761536951, 
    0.602413724733, 
    0.462155539625, 
    0.487964030916, 
    0.440976024257, 
    0.414337388691, 
    0.378996487974
    ]  
improvement2 = [
    0.62404,
    0.34791,
    0.45274,
    0.31445,
    0.36108,
    0.30780,
    0.32434,
    0.26444]

#pylab.plot(classes, trainerr, 'ro--', label='Train accuracy')
#pylab.plot(classes, testerr, 'g^--', label='Test accuracy')
#pylab.plot(classes, improvement, 'ys-.', label='Improvement')
#pylab.plot(classes, F1, 'p:', label='F1')
#pylab.legend(loc='upper right')
#pylab.xlabel('Number of classes')
#pylab.ylabel('Fraction')
#pylab.show()

pylab.plot(classes, train_accuracy, 'ro--', label='Train accuracy')
pylab.plot(classes, test_accuracy, 'g^--', label='Test accuracy')
pylab.plot(classes, improvement2, 'ys-.', label='Improvement')
pylab.plot(classes, f1_score, 'p:', label='F1')
pylab.legend(loc='upper right')
pylab.xlabel('Number of classes')
pylab.ylabel('Fraction')
pylab.show()