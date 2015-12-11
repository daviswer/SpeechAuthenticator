Hello!

First, the auxiliary files:

WAVreader.py is a class, initialized on an array of data files and a labeling scheme, which reads in the audio, performs MFCC and delta calculations, applies the labeling scheme, and stores the data sets which the machine learning systems use to train and test. 

SNN.py is the neural tensor network used for both neural models. Training/testing it on a data set requires extracting the appropriate data set from WAVreader. 

QLing.py is our Q-learner neural network implementation. Again, training/testing it requires extracting the appropriate data set from WAVreader. 

Command to run each script: python <filename>

Next, the testing scripts:

SVMtest.ipynb is an ipython notebook file that uses WAVreader on a list of files, extracts the appropriate dataset, trains an SVM, outputs a confusion matrix, and extracts the appropriate testing metrics. 

SOFTMAX TESTER.ipynb is an ipython notebook that uses WAVreader on a list of files, extracts the appropriate dataset, and trains a neural network model on it with the specified parameters. Outputs a confusion matrix, and extracts the appropriate testing metrics. 

RECORDER.ipynb is an ipython notebook that records audio and saves it to a wav file. 

QLEARNING.ipynb is an ipython notebook which extends SVMtest by not only training the SVM and extracting accuracy metrics, but also training an array of Q-Learner networkers using that SVM. Each Q-Learner is trained on a different resampling penalty parameter value. 

LIVEDEMO.ipynb is the ipython implementation of a simple real-time audio recorder and classifier based on SVM.

Misc.

LIVEDEMO.py is the script version of the real-time classifier. 

plots.py was used to generate some of the plots appearing in the report. 
