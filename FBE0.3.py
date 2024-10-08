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
from sklearn.cluster import DBSCAN, KMeans
import soundfile as sf


DATA_F = 'C:/Users/markm/Desktop/New_Choir_Dataset'
DATA_TEST_SINGLE = 'C:/Users/markm/Desktop/New_Choir_Dataset/Segment_9_Choir.wav'
S_R = 44100 # Refactor so it would check the datasets sample rate, if there are different sample rates, resample into the lower one

FFT_N = 2048
HOP_L = FFT_N // 4
MEL_N = 128
MFCC_N = 13 

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

# here mfcc should be centroids from clusters
def process_envelope(mfcc, n_mels = MEL_N, n_fft = FFT_N, sr = S_R):
    mel_spectrogram = mfcc_to_mel(mfcc)
    stft = librosa.feature.inverse.mel_to_stft(mel_spectrogram, n_fft=n_fft)
    return mel_spectrogram, stft

def random_sample_frames(mfcc, num_samples=10000):
    np.random.shuffle(mfcc.T)
    samples = mfcc
    return samples[:,:num_samples].T

def process_data(data):
    mfcc_list = []
    for root, dirs, files in os.walk(data):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                mfcc = extract_mfcc(file_path)
                sampled = random_sample_frames(mfcc)
                mfcc_list.append(sampled)
    return np.vstack(mfcc_list)

def dbscan_cluster_number(data):
    dbscan = DBSCAN(eps=3, min_samples=10)
    dbscan.fit(data)
    labels = dbscan.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, num_clusters


    

    



