
#Neuron Spike Classifier:
University Coursework project for my 'Computational intelligence module' 
Classified 'spike' signals produced by neurons into 5 catagories using ML techniques. Models are trained on dataset D1 (containing minimal noise)
Datasets D2 to D6 contained increasing levels of noise. FFT and wavelet filtering were found to be particularly effective at removing this noise. 


## Required Libraries:
-tensorflow
-keras
-sklearn
-scipy
-pywt
-matplotlib

## Instructions:
-Run neuronSpikeTrainer to train models on D1
-Run neuronSpikeClassifier to classify D2-D6
-Classifications saved in outoutDataV2 File. Classificationes contain spike index and classification 1-6
