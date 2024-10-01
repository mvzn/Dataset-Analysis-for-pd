# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:14:03 2024

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

def random_sample_frames(mfcc, num_samples=10000, frame_size=MEL_N):
    num_frames = mfcc.shape[1]
    
    # Ensure there are enough frames to sample from
    if num_frames == 0:
        raise ValueError("MFCC data contains no frames to sample from.")

    # If there are fewer frames than requested, adjust the number of samples
    if num_samples > num_frames:
        print(f"Warning: Number of requested samples ({num_samples}) exceeds the available frames ({num_frames}). Adjusting to {num_frames} samples.")
        num_samples = min(num_samples, num_frames)

    # Sample random frames from the MFCC matrix
    sampled_frames = np.array([mfcc[:, random.randint(0, num_frames-1)] for _ in range(num_samples)])

    return sampled_frames

def find_clusters_with_dbscan(data):
    dbscan = DBSCAN(eps=2.5, min_samples=10)
    dbscan.fit(data)
    labels = dbscan.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return num_clusters, labels

def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=300, n_init=10)
    kmeans.fit(data)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers, labels

def process_dataset(folder_path, num_samples=10000):
    mfcc_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                mfcc = extract_mfcc(file_path)
                sampled_frames = random_sample_frames(mfcc, num_samples=num_samples // len(files))
                mfcc_list.append(sampled_frames)
    
    return np.vstack(mfcc_list)

def process_envelope(mfcc, n_mels = MEL_N, n_fft = FFT_N, sr = S_R):
    mel_spectrogram = mfcc_to_mel(mfcc)
    
    stft = librosa.feature.inverse.mel_to_stft(mel_spectrogram, n_fft=n_fft)

    return mel_spectrogram, stft

mfcc_data = process_dataset(DATA_F)

num_clusters, dbscan_labels = find_clusters_with_dbscan(mfcc_data)
print(f'Estimated number of clusters: {num_clusters}')

centroids, kmeans_labels = kmeans_clustering(mfcc_data, num_clusters)
print(f'Cluster centroids:\n{centroids}')

mel_spec, stft = process_envelope(centroids, n_mels=num_clusters)






