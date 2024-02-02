import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import scipy.io as spio 
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import find_peaks
import pywt
mat = spio.loadmat('datasets/D1.mat', squeeze_me=True)

#Filter signal to remove low and high frequency noise for processing by spike detector
def butterBandpassFilter(data, lowcut, highcut, samplingRate, order=4):
    nyquist = 0.5 * samplingRate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(N=order, Wn=[low, high], btype='band', output='ba')
    filteredData = filtfilt(b, a, data) #Use filtfilt to prevent phase shift

    return filteredData

#Filter signal to remove low frequency noise for processing by spike detector
def butterLowpassFilter(data, cutoffFreq, samplingRate, order=4):
    nyquist = 0.5 * samplingRate
    cutoff = cutoffFreq / nyquist
    b, a = butter(N=order, Wn=cutoff, btype='low', output='ba')
    filteredData = filtfilt(b, a, data) #Use filtfilt to prevent phase shift

    return filteredData


#Wavelet filter produces good results for filtering signals D2 & 3 before passing through model
def waveletDenoise(data, threshold, wavelet= 'db1', level= 6):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffsThresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    denoisedData = pywt.waverec(coeffsThresholded, wavelet)
    return denoisedData

#Identification of spikes
def spikeClassifierD2():
    mat = spio.loadmat('datasets/D1.mat', squeeze_me=True)
    d = mat['d']
    givenIndex = mat['Index']
    classLabels = mat['Class']
    index = []
    newClasses = []
    newIndex = []
    spikeList = []


    # Add noise to training signal to replicate noise in expected data
    mean = 0  # Mean of noise
    stdDev = 0.25  # Standard deviation of noise
    noise = np.random.normal(mean, stdDev, len(d))
    noisySignal = d + noise

    #Different filtering required for detecting spikes and signal passed to CNN
    bandFilteredSig= butterBandpassFilter(noisySignal, 10, 500, 25000) #Filter for spike detection  
    waveFilteredSig=waveletDenoise(noisySignal, 0.7) #Filter signal passed into CNN
    differential = np.gradient(bandFilteredSig) #Peaks chosen by looking at peaks in the gradient of the signal
    peaks, _ = find_peaks(differential, height= 0.03, distance = 25)

    i = 0
    while i <= len(peaks)-1:
        if bandFilteredSig[peaks[i]] < 0:
            peaks= np.delete(peaks, i)  #Delete any incorrect 'negative' peaks 
        else:
            i = i+1
    
    for i in range(len(peaks)):
        index.append(peaks[i]-10) #Start indexes of spikes       

  #Find classes associated with each spike by comparing spike indexes given in D1 to spikes found
    for i in range(len(givenIndex)):
        for x in index:
            if givenIndex[i] - 25 <= x <= givenIndex[i] + 25:  
                newClasses.append(classLabels[i])
                newIndex.append(x)        
    for i in range(len(newIndex)): 
        spikeList.append(waveFilteredSig[newIndex[i]:newIndex[i]+50])     


    spikeData = np.array(spikeList)
    classifications = np.array(newClasses) 
    return spikeData, classifications
def spikeClassifierD3():
    mat = spio.loadmat('datasets/D1.mat', squeeze_me=True)
    d = mat['d']
    givenIndex = mat['Index']
    classLabels = mat['Class']
    index = []
    newClasses = []
    newIndex = []
    spikeList = []


    # Add noise to training signal to replicate noise in expected data
    mean = 0  # Mean of noise
    stdDev = 0.5  # Standard deviation of noise
    noise = np.random.normal(mean, stdDev, len(d))
    noisySignal = d + noise

    #Different filtering required for detecting spikes and signal passed to CNN
    bandFilteredSig= butterBandpassFilter(noisySignal, 10, 500, 25000) #Filter for spike detection  
    waveFilteredSig=waveletDenoise(noisySignal, 1.1) #Filter signal passed into CNN
    differential = np.gradient(bandFilteredSig) #Peaks chosen by looking at peaks in the gradient of the signal
    peaks, _ = find_peaks(differential, height= 0.035, distance = 25)

    i = 0
    while i <= len(peaks)-1:
        if bandFilteredSig[peaks[i]] < 0:
            peaks= np.delete(peaks, i)  #Delete any incorrect 'negative' peaks 
        else:
            i = i+1
    
    for i in range(len(peaks)):
        index.append(peaks[i]-10) #Start indexes of spikes       

  #Find classes associated with each spike by comparing spike indexes given in D1 to spikes found
    for i in range(len(givenIndex)):
        for x in index:
            if givenIndex[i] - 25 <= x <= givenIndex[i] + 25:  
                newClasses.append(classLabels[i])
                newIndex.append(x)        
    for i in range(len(newIndex)): 
        spikeList.append(waveFilteredSig[newIndex[i]:newIndex[i]+50])     


    spikeData = np.array(spikeList)
    classifications = np.array(newClasses) 
    return spikeData, classifications
def spikeClassifierD4():
    mat = spio.loadmat('datasets/D1.mat', squeeze_me=True)
    d = mat['d']
    givenIndex = mat['Index']
    classLabels = mat['Class']
    index = []
    newClasses = []
    newIndex = []
    spikeList = []


    # Add noise to training signal to replicate noise in expected data
    mean = 0  # Mean of noise
    stdDev = 0.5  # Standard deviation of noise
    noise = np.random.normal(mean, stdDev, len(d))
    noisySignal = d + noise
    #Different filtering required for detecting spikes and signal passed to KNN
    bandFilteredSig= butterBandpassFilter(noisySignal, 10, 500, 25000) #Filter for spike detection  
    lowFilteredSignal= butterLowpassFilter(noisySignal, 1000, 25000, order=4) #Filter signal passed to cnn 
    differential = np.gradient(bandFilteredSig)
    peaks, _ = find_peaks(differential, height= 0.04, distance = 25)
    index = []
    newClasses = []
    newIndex = []
    spikeList = []

    i = 0
    while i <= len(peaks)-1:
        if bandFilteredSig[peaks[i]] < 0:
            peaks= np.delete(peaks, i) #Delete any incorrect 'negative' peaks 
        else:
            i = i+1
    
    for i in range(len(peaks)):
        index.append(peaks[i]-10)  #Start indexes of spikes  

#Find classes associated with each spike by comparing spike indexes given in D1 to spikes found
    for i in range(len(givenIndex)): 
        for x in index:
            if givenIndex[i] - 25 <= x <= givenIndex[i] + 25:
                newClasses.append(classLabels[i])
                newIndex.append(x)        
    for i in range(len(newIndex)): 
        spikeList.append(lowFilteredSignal[newIndex[i]:newIndex[i]+50])     

    spikeData = np.array(spikeList)
    classifications = np.array(newClasses)
    return spikeData, classifications

def spikeClassifierD5():
    mat = spio.loadmat('datasets/D1.mat', squeeze_me=True)
    d = mat['d']
    givenIndex = mat['Index']
    classLabels = mat['Class']
    mean = 0  
    stdDev = 1.3  
    noise = np.random.normal(mean, stdDev, len(d))
    noisySignal = d + noise

    noisySignal = butterBandpassFilter(noisySignal, 100, 12000, 250000)
    noisySignalNormalised = np.expand_dims(noisySignal, axis=-1)
    fftResult = np.fft.fft(noisySignalNormalised)
    threshold = 4
    attenuation_factor =  0.3  
    fftResultFiltered = np.where(np.abs(fftResult) < threshold, fftResult * attenuation_factor, fftResult)
    denoisedSignal = np.fft.ifft(fftResultFiltered).real
    denoisedSignal= denoisedSignal.flatten()

    peaks, _ = find_peaks(denoisedSignal, height= 0.7, distance = 25)
    index = []
    newClasses = []
    newIndex = []
    spikeList = []
    

    i = 0
    while i <= len(peaks)-1:
        if denoisedSignal[peaks[i]] < 0:
            peaks= np.delete(peaks, i)
        else:
            i = i+1
    
    for i in range(len(peaks)):
        index.append(peaks[i]-25)      

    for i in range(len(givenIndex)):
        for x in index:
            if givenIndex[i] - 25 <= x <= givenIndex[i] + 25:
                newClasses.append(classLabels[i])
                newIndex.append(x)        
    for i in range(len(newIndex)): 
        spikeList.append(denoisedSignal[newIndex[i]:newIndex[i]+50])     

    spikeData = np.array(spikeList)
    classifications = np.array(newClasses)
    return spikeData, classifications

def spikeClassifierD6():
    mat = spio.loadmat('datasets/D1.mat', squeeze_me=True)
    d = mat['d']
    givenIndex = mat['Index']
    classLabels = mat['Class']
    mean = 0  
    stdDev = 2.7  
    noise = np.random.normal(mean, stdDev, len(d))
    noisySignal = d + noise

    noisySignal = butterBandpassFilter(noisySignal, 100, 12000, 250000)
    noisySignalNormalised = np.expand_dims(noisySignal, axis=-1)
    fftResult = np.fft.fft(noisySignalNormalised)
    threshold = 4
    attenuation_factor =  0.3  
    fftResultFiltered = np.where(np.abs(fftResult) < threshold, fftResult * attenuation_factor, fftResult)
    denoisedSignal = np.fft.ifft(fftResultFiltered).real
    denoisedSignal= denoisedSignal.flatten()

    peaks, _ = find_peaks(denoisedSignal, height= 0.75, distance = 25)
    index = []
    newClasses = []
    newIndex = []
    spikeList = []
    

    i = 0
    while i <= len(peaks)-1:
        if denoisedSignal[peaks[i]] < 0:
            peaks= np.delete(peaks, i)
        else:
            i = i+1
    
    for i in range(len(peaks)):
        index.append(peaks[i]-25)      

    for i in range(len(givenIndex)):
        for x in index:
            if givenIndex[i] - 25 <= x <= givenIndex[i] + 25:
                newClasses.append(classLabels[i])
                newIndex.append(x)        
    for i in range(len(newIndex)): 
        spikeList.append(denoisedSignal[newIndex[i]:newIndex[i]+50])     

    spikeData = np.array(spikeList)
    classifications = np.array(newClasses)
    return spikeData, classifications


def CNND2():
    spikeData, classifications = spikeClassifierD2()
    # Split data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(spikeData, classifications, test_size=0.2, random_state=42)

    # Reshape the data for CNN input, given spikes have a length of 50
    xTrain = xTrain.reshape((xTrain.shape[0], 50, 1))
    xTest = xTest.reshape((xTest.shape[0], 50, 1))
    inputShape = (50, 1)  
    # Define CNN model
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))  #5 classes, 0th class is ignored 

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model on D1 signal
    model.fit(xTrain, yTrain, epochs=50, validation_data=(xTest, yTest))
    testLoss, testAcc = model.evaluate(xTest, yTest)
    print(testAcc)
    model.save('models/cnn2.h5')
def CNND3():
    spikeData, classifications = spikeClassifierD3()
    # Split data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(spikeData, classifications, test_size=0.2, random_state=42)

    # Reshape the data for CNN input, given spikes have a length of 50
    xTrain = xTrain.reshape((xTrain.shape[0], 50, 1))
    xTest = xTest.reshape((xTest.shape[0], 50, 1))
    inputShape = (50, 1)  
    # Define CNN model
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))  #5 classes, 0th class is ignored 

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model on D1 signal
    model.fit(xTrain, yTrain, epochs=50, validation_data=(xTest, yTest))
    testLoss, testAcc = model.evaluate(xTest, yTest)
    print(testAcc)
    model.save('models/cnn3.h5')
def KNND4():
    spikeData, classifications = spikeClassifierD4()
 
    # Split the data into training and testing sets

    # Reshape the data to have two dimensions (assuming the waveforms have a length of 50)
    inputShape = (50,)  # Adjust if your waveforms have a different length
    spikeData_2d = spikeData.reshape((spikeData.shape[0], -1))  
    # Split the data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(spikeData_2d, classifications, test_size=0.2, random_state=42)

    # Reshape the data for CNN input (assuming the waveforms have a length of 50)
    inputShape = (50, 1)  # Adjust if your waveforms have a different length
    knn_classifier = KNeighborsClassifier(n_neighbors=10)  # You can adjust the number of neighbors
    knn_classifier.fit(xTrain, yTrain)
    predictedLabels = knn_classifier.predict(xTest)
    model_filename = 'knn4.joblib'
    joblib.dump(knn_classifier, model_filename)

def CNND5():
    spikeData, classifications = spikeClassifierD5()
    # Split data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(spikeData, classifications, test_size=0.2, random_state=42)

    # Reshape the data for CNN input, given spikes have a length of 50
    xTrain = xTrain.reshape((xTrain.shape[0], 50, 1))
    xTest = xTest.reshape((xTest.shape[0], 50, 1))
    inputShape = (50, 1)  
    # Define CNN model
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))  #5 classes, 0th class is ignored 

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model on D1 signal
    model.fit(xTrain, yTrain, epochs=50, validation_data=(xTest, yTest))
    testLoss, testAcc = model.evaluate(xTest, yTest)
    print(testAcc)
    model.save('models/cnn5.h5')

def CNND6():
    spikeData, classifications = spikeClassifierD6()
    # Split data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(spikeData, classifications, test_size=0.2, random_state=42)

    # Reshape the data for CNN input, given spikes have a length of 50
    xTrain = xTrain.reshape((xTrain.shape[0], 50, 1))
    xTest = xTest.reshape((xTest.shape[0], 50, 1))
    inputShape = (50, 1)  
    # Define CNN model
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=inputShape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(6, activation='softmax'))  #5 classes, 0th class is ignored 

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model on D1 signal
    model.fit(xTrain, yTrain, epochs=50, validation_data=(xTest, yTest))
    testLoss, testAcc = model.evaluate(xTest, yTest)
    print(testAcc)
    model.save('models/cnn6.h5')


#Run classifier
CNND2()
CNND3()
KNND4()
CNND5()
CNND6()

