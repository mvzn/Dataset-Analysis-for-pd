# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:17:17 2024

@author: markm
"""

import os
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler
import soundfile as sf


CHOIR_F = 'C:/Users/markm/Desktop/New_Choir_Dataset'
CHOIR_S = 'C:/Users/markm/Desktop/New_Choir_Dataset/Segment_9_Choir.wav'
NASA_F = 'C:/Users/markm/Desktop/NASA_Dataset/'
NASA_S = 'C:/Users/markm/Desktop/NASA_Dataset/Apollo 11 Mission Audio - Day 1.wav'
S_R = 44100 # Refactor so it would check the datasets sample rate, if there are different sample rates, resample into the lower one

FFT_N = 1024
HOP_L = FFT_N // 4
MEL_N = 128
MFCC_N = 20 

scaler = StandardScaler()

# Functions
def extract_mfcc(file_path, sr = S_R, n_fft = FFT_N, n_mels = MEL_N, n_mfcc = MFCC_N, hop_length = HOP_L):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft = n_fft, n_mels = n_mels, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc

def mel_filter(sr = S_R, n_fft = FFT_N, n_mels = MEL_N):
    mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return mel_filter

def mfcc_to_mel(mfcc, n_mels = MEL_N):
    mel_mfcc = librosa.feature.inverse.mfcc_to_mel(mfcc, n_mels=n_mels)
    return mel_mfcc

def random_sample_frames(mfcc, num_samples=500):
    samples = mfcc.T
    np.random.shuffle(samples)
    samples = samples[:num_samples,:]
    print(samples.shape, num_samples)
    return samples
#[:,:num_samples].T

def standardize(data):
    return scaler.fit_transform(data)

def unstandardize(data):
    return scaler.inverse_transform(data)

def normalize(data):
    return librosa.util.normalize(data)

def dbscan_cluster_number(data):
    dbscan = DBSCAN(eps=0.01)
    labels = dbscan.fit_predict(data) 
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return num_clusters, labels

def hdbscan_clustering(data):
    hdb = HDBSCAN()
    labels = hdb.fit_predict(data)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return num_clusters, labels

def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters = num_clusters, n_init=50)
    kmeans.fit(data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_ 
    return cluster_centers, labels

# main processes

def process_data(data, num_samples=40000):
    mfcc_list = []
    for root, dirs, files in os.walk(data):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                mfcc = extract_mfcc(file_path)
                print(file_path)
                silence = np.where(mfcc[0, :] < min(mfcc[0,:]) * 0.7)
                new_mfcc = np.delete(mfcc, silence, 1)
                print(new_mfcc.shape)
                sampled = random_sample_frames(new_mfcc, num_samples=num_samples // len(files))
                mfcc_list.append(sampled)
    return np.vstack(mfcc_list)

# here mfcc should be centroids from clusters
def process_envelope(mfcc, n_mels = MEL_N, n_fft = FFT_N, sr = S_R):
    mel_spectrogram = mfcc_to_mel(mfcc.T)
    stft = librosa.feature.inverse.mel_to_stft(mel_spectrogram, n_fft=n_fft)
    print(mel_spectrogram.shape, stft.shape)
    return mel_spectrogram, stft


# calls

DATA_F = NASA_F

mfcc_data = process_data(DATA_F)

s_mfcc_data = standardize(mfcc_data)

#num_clusters, dbscan_labels = dbscan_cluster_number(s_mfcc_data)
num_clusters, hdbscan_labels = hdbscan_clustering(s_mfcc_data)
print(f'Estimated number of clusters: {num_clusters}')

centroids, kmeans_labels = kmeans_clustering(s_mfcc_data, num_clusters)
print(f'Cluster centroids:\n{centroids}')

mel_spec, stft = process_envelope(unstandardize(centroids))

abs_stft = np.abs(stft)

plt.plot(stft[:200,:])

