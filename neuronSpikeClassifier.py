import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io as spio 
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import find_peaks, peak_widths
from scipy import signal
import joblib
import pywt

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
#Generate .mat files with classification outputs 
def generateMat(predictedLabels,index, fileName):
    print(len(predictedLabels)) #Check data is present
    dataDict = {'Index': index, 'Class': predictedLabels}
    filePath = 'outputDataV3/' + fileName
    spio.savemat(filePath, dataDict)

#Identify spikes and index they occur at 
def spikeDetectorD2():
    mat = spio.loadmat('datasets/D2.mat', squeeze_me=True)
    sig = mat['d']
    sig2 =  mat['d']
    #Different filtering required for detecting spikes and signal passed to CNN
    bandFilteredSig = butterBandpassFilter(sig, 100, 500, 25000) #Filter for spike detection  
    waveFilteredSig = waveletDenoise(sig2, 0.7)  #Filter signal passed into CNN
    spikeList = []
    differential = np.gradient(bandFilteredSig)  #Peaks chosen by looking at peaks in the gradient of the signal
    peaks, _ = find_peaks(differential, height= 0.03, distance = 25)
    index = []
    
    i = 0
    while i <= len(peaks)-1:
        if bandFilteredSig[peaks[i]] < 0:
            peaks= np.delete(peaks, i) #Delete any incorrect 'negative' peaks 
        else:
            i = i+1
    for i in range(len(peaks)):   
        spikeList.append(waveFilteredSig[peaks[i]-10:peaks[i]+40]) #Array of 'windows' containing each spike 
        index.append(peaks[i]-10) #Start indexes of spikes 

    spikeData = np.array(spikeList)
    
    return spikeData,  index
#Execute classifier to D2 

def spikeDetectorD3():
    mat = spio.loadmat('datasets/D3.mat', squeeze_me=True)
    sig = mat['d']
    sig2 =  mat['d']
    #Different filtering required for detecting spikes and signal passed to CNN
    bandFilteredSig = butterBandpassFilter(sig, 100, 500, 25000) #Filter for spike detection  
    waveFilteredSig = waveletDenoise(sig2, 1.1)  #Filter signal passed into CNN
    spikeList = []
    differential = np.gradient(bandFilteredSig)  #Peaks chosen by looking at peaks in the gradient of the signal
    peaks, _ = find_peaks(differential, height= 0.025, distance = 25)
    index = []
    
    i = 0
    while i <= len(peaks)-1:
        if bandFilteredSig[peaks[i]] < 0:
            peaks= np.delete(peaks, i) #Delete any incorrect 'negative' peaks 
        else:
            i = i+1
    for i in range(len(peaks)):   
        spikeList.append(waveFilteredSig[peaks[i]-10:peaks[i]+40]) #Array of 'windows' containing each spike 
        index.append(peaks[i]-10) #Start indexes of spikes 

    spikeData = np.array(spikeList)
    
    return spikeData,  index
#Execute classifier to D2 
def spikeDetectorD4():
    mat = spio.loadmat('datasets/D4.mat', squeeze_me=True)
    sig = mat['d']
    sig2 =  mat['d']
    #Different filtering required for detecting spikes and signal passed to CNN
    bandFilteredSig = butterBandpassFilter(sig, 100, 500, 25000) #Filter for spike detection  
    lowFilteredSig = butterLowpassFilter(sig2, 1000, 25000)  #Filter signal passed into CNN- wavelet filtering found to cause too much artifacting
    spikeList = []
    differential = np.gradient(bandFilteredSig)  #Peaks chosen by looking at peaks in the gradient of the signal
    peaks, _ = find_peaks(differential, height= 0.04, distance = 25)
    index = []
    
    i = 0
    while i <= len(peaks)-1:
        if bandFilteredSig[peaks[i]] < 0:
            peaks= np.delete(peaks, i) #Delete any incorrect 'negative' peaks 
        else:
            i = i+1
    for i in range(len(peaks)):   
        spikeList.append(lowFilteredSig [peaks[i]-10:peaks[i]+40]) #Array of 'windows' containing each spike 
        index.append(peaks[i]-10) #Start indexes of spikes 

    spikeData = np.array(spikeList)
    
    return spikeData,  index


def spikeDetectorD5():
    mat = spio.loadmat('datasets/D5.mat', squeeze_me=True)
    sig = mat['d']
    threshold = 4
    attenuationFactor =  0.3 
    index = []
    spikeList = []

    bandFilteredSig = butterBandpassFilter(sig, 100, 12000, 250000)  #Initially remove lowe freq noise and (some) high freq
    sigExt = np.expand_dims(bandFilteredSig, axis=-1) 
    fftResult = np.fft.fft(sigExt) #Perform fast fourier transform to convert to freq domain
    fftResultFiltered = np.where(np.abs(fftResult) < threshold, fftResult * attenuationFactor, fftResult)   #Remove low amplitude (noise) freq components from signal
    denoisedSignal = np.fft.ifft(fftResultFiltered).real  #Convert back to freq domain
    denoisedSignal= denoisedSignal.flatten()
    peaks, _ = find_peaks(denoisedSignal, height= 0.75, distance = 25) #Find peaks on  denoised signal

    
    i = 0
    while i <= len(peaks)-1:
        if denoisedSignal[peaks[i]] < 0:
            peaks= np.delete(peaks, i)  #Delete any incorrect 'negative' peaks 
        else:
            i = i+1
    for i in range(len(peaks)):
        spikeList.append(denoisedSignal[peaks[i]-25:peaks[i]+25])
        index.append(peaks[i]-25) #Start indexes of spikes 

    spikeData = np.array(spikeList)

    #plotPeakGraphs(denoised_signal, sig2, originalSig,  index, ind= [])

    return spikeData, index


def spikeDetectorD6():
    mat = spio.loadmat('datasets/D6.mat', squeeze_me=True)
    sig = mat['d']
    threshold = 4
    attenuationFactor =  0.3 
    index = []
    spikeList = []

    bandFilteredSig = butterBandpassFilter(sig, 100, 12000, 250000)  #Initially remove lowe freq noise and (some) high freq
    sigExt = np.expand_dims(bandFilteredSig, axis=-1) 
    fftResult = np.fft.fft(sigExt) #Perform fast fourier transform to convert to freq domain
    fftResultFiltered = np.where(np.abs(fftResult) < threshold, fftResult * attenuationFactor, fftResult)   #Remove low amplitude (noise) freq components from signal
    denoisedSignal = np.fft.ifft(fftResultFiltered).real  #Convert back to freq domain
    denoisedSignal= denoisedSignal.flatten()
    peaks, _ = find_peaks(denoisedSignal, height= 0.75, distance = 25) #Find peaks on  denoised signal

    
    i = 0
    while i <= len(peaks)-1:
        if denoisedSignal[peaks[i]] < 0:
            peaks= np.delete(peaks, i)  #Delete any incorrect 'negative' peaks 
        else:
            i = i+1
    for i in range(len(peaks)):
        spikeList.append(denoisedSignal[peaks[i]-25:peaks[i]+25])
        index.append(peaks[i]-25) #Start indexes of spikes 

    spikeData = np.array(spikeList)

    #plotPeakGraphs(denoised_signal, sig2, originalSig,  index, ind= [])

    return spikeData, index


def D2Classify():
    spikeDataD2, index = spikeDetectorD2() #Get spikes and indexes that they occur at
    spikeDataD2.reshape((spikeDataD2.shape[0], 50, 1))
    cnnD2 = tf.keras.models.load_model('models/cnn2.h5') #Load CNN model 
    predictions = cnnD2.predict(spikeDataD2) #Run spike data through CNN
    predictedLabels = np.argmax(predictions, axis=1) 
    generateMat(predictedLabels, index, "D2.mat") #Save labels to file  

def D3Classify():
    spikeDataD3, index = spikeDetectorD3() #Get spikes and indexes that they occur at
    spikeDataD3.reshape((spikeDataD3.shape[0], 50, 1))
    cnnD3 = tf.keras.models.load_model('models/cnn3.h5') #Load CNN model 
    predictions = cnnD3.predict(spikeDataD3) #Run spike data through CNN
    predictedLabels = np.argmax(predictions, axis=1) 
    generateMat(predictedLabels, index, "D3.mat") #Save labels to file  

def D4Classify():
    spikeDataD4, index = spikeDetectorD4()
    spikeDataD4.reshape((spikeDataD4.shape[0], 50, 1))   #Get spikes and indexes that they occur at
    spikeData_2d = spikeDataD4.reshape((spikeDataD4.shape[0], -1))
    loaded_knn_model = joblib.load('knn4.joblib') #Load KNN model
    predictedLabels = loaded_knn_model.predict(spikeData_2d) #Run spike data through KNN
    generateMat(predictedLabels, index, 'D4.mat') #Save labels to file  

def D5Classify():
    spikeDataD5, index = spikeDetectorD5() #Get spikes and indexes that they occur at
    spikeDataD5.reshape((spikeDataD5.shape[0], 50, 1))
    cnnD5 = tf.keras.models.load_model('models/cnn5.h5') #Load CNN model 
    predictions = cnnD5.predict(spikeDataD5) #Run spike data through CNN
    predictedLabels = np.argmax(predictions, axis=1) 
    generateMat(predictedLabels, index, "D5.mat") #Save labels to file  

def D6Classify():
    spikeDataD6, index = spikeDetectorD6() #Get spikes and indexes that they occur at
    spikeDataD6.reshape((spikeDataD6.shape[0], 50, 1))
    cnnD6 = tf.keras.models.load_model('models/cnn6.h5') #Load CNN model 
    predictions = cnnD6.predict(spikeDataD6) #Run spike data through CNN
    predictedLabels = np.argmax(predictions, axis=1) 
    generateMat(predictedLabels, index, "D6.mat") #Save labels to file  

D2Classify()
D3Classify()
D4Classify()
D5Classify()
D6Classify()
