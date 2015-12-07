import pylab 

classes = [2,3,4,5,6,7,8,9]
#NTN
train_accuracyNTN = [.965,.805,.628,.513,.375,.334,.199,.182]
test_accuracyNTN = [.804,.588,.466,.394,.301,.268,.161,.148]
improvementNTN = [.608,.382,.288,.243,.161,.146,.041,.042]
f1_scoreNTN = [.804,.586,.461,.390,.325,.264,.197,.173]

#SNN
train_accuracySNN= [0.866541353383,0.739687055477,0.642244086247,0.589601046436,0.488643999462,0.463743315508,0.46146738126,0.385661110212]
test_accuracySNN = [0.863387978142, 0.722332670643,0.6373840898,0.553012967201,0.458790347853,0.438123752495,0.418930573523,0.357682144205]
f1_scoreSNN = [0.867283703166,0.725822367694,0.641356334167,0.572167333243,0.503835275987,0.454756642688,0.425891148186,0.372475691601]  
improvementSNN = [0.72678,0.58350,0.51651,0.44127,0.35055,0.34448,0.33592,0.27739]

#SVM
train_accuracySVM = [0.883458646617,0.742816500711,0.669039145907,0.630640941792,0.592393495498,0.562994652406,0.525318795047,0.478151804499]
test_accuracySVM = [0.874316939891,0.730947647449,0.666178623719,0.615179252479,0.556878721404,0.522704590818,0.489003880983,0.445828614572]
f1_scoreSVM = [0.874420950896,0.731517308737,0.666579117388,0.615398724056,0.555470149322,0.470622788513,0.485977114985,0.443577893463]  
improvementSVM = [0.74863,0.59642,0.55490,0.51897,0.46825,0.44316,0.41600,0.37656]

SNN = pylab.figure(1)
with pylab.style.context('fivethirtyeight'):
	pylab.plot(classes, train_accuracySNN,label='Train accuracy')
	pylab.plot(classes, test_accuracySNN,label='Test accuracy')
	pylab.plot(classes, improvementSNN, label='Improvement')
	pylab.plot(classes, f1_scoreSNN, label='F1')
	pylab.legend(loc='upper right',prop={'size':9})
	pylab.xlabel('Number of classes')
	pylab.ylabel('Fraction')
	pylab.title('Plot of Softmax Neural Network')
SNN.show()

NTN = pylab.figure(2)
with pylab.style.context('fivethirtyeight'):
	pylab.plot(classes, train_accuracyNTN,label='Train accuracy')
	pylab.plot(classes, test_accuracyNTN,label='Test accuracy')
	pylab.plot(classes, improvementNTN, label='Improvement')
	pylab.plot(classes, f1_scoreNTN, label='F1')
	pylab.legend(loc='upper right',prop={'size':9})
	pylab.xlabel('Number of classes')
	pylab.ylabel('Fraction')
	pylab.title('Plot of Neural Tensor Network')
pylab.show()

SVM = pylab.figure(3)
with pylab.style.context('fivethirtyeight'):
	pylab.plot(classes, train_accuracySVM,label='Train accuracy')
	pylab.plot(classes, test_accuracySVM,label='Test accuracy')
	pylab.plot(classes, improvementSVM, label='Improvement')
	pylab.plot(classes, f1_scoreSVM, label='F1')
	pylab.legend(loc='upper right',prop={'size':9})
	pylab.xlabel('Number of classes')
	pylab.ylabel('Fraction')
	pylab.title('Plot of Support Vector Machine')
pylab.show()

compare  = pylab.figure(4)
with pylab.style.context('fivethirtyeight'):
	pylab.subplot(2, 1, 1)
	pylab.plot(classes, test_accuracySNN, label='Improvement for SNN')
	pylab.plot(classes, test_accuracyNTN, label='Improvement for NTN')
	pylab.plot(classes, test_accuracySVM, label='Improvement for SVM')
	pylab.legend(loc='upper right',prop={'size':9})
	pylab.ylabel('Fraction')
	pylab.title('Comparing test accuracy of NTN, SNN, SVM')
	pylab.subplot(2, 1, 2)
	pylab.plot(classes, improvementSNN, label='Improvement for SNN')
	pylab.plot(classes, improvementNTN, label='Improvement for NTN')
	pylab.plot(classes, improvementSVM, label='Improvement for SVM')
	pylab.legend(loc='upper right',prop={'size':9})
	pylab.xlabel('Number of classes')
	pylab.ylabel('Fraction')
	pylab.title('Comparing improvements of NTN, SNN, SVM')
pylab.show()