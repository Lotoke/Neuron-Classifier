# Neuron Spike Classifier

University coursework project for the 'Computational Intelligence' module. The project involves classifying 'spike' signals produced by neurons into 5 categories using machine learning techniques. Models are trained on dataset D1, which contains minimal noise. Datasets D2 to D6 contain increasing levels of noise. FFT and wavelet filtering were found to be particularly effective at removing this noise.

## Required Libraries:

- tensorflow
- keras
- sklearn
- scipy
- pywt
- matplotlib

## Instructions:

- Run `neuronSpikeTrainer` to train models on dataset D1.
- Run `neuronSpikeClassifier` to classify datasets D2 to D6.
- Classifications are saved in the `outputDataV2` file. Classifications contain spike index and classification values ranging from 1 to 6.
