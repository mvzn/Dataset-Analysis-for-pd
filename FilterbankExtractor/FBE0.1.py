# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:36:45 2024

@author: markm

"""

import os
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from librosa.feature.inverse import mfcc_to_mel, mel_to_stft, filters
import soundfile as sf

# Set
dataset_folder = 'C:/Users/markm/Desktop/New_Choir_Dataset'
sample_rate = 44100

DURATION = 2         # Duration of the noise signal in seconds
N_FFT = 2048         # FFT size
HOP_LENGTH = N_FFT // 4  # Hop length for STFT
N_MELS = 40  


# Ready

# Function to load audio and extract MFCC
def extract_mfcc(file_path, sr=sample_rate, n_mfcc=N_MELS, hop_length=HOP_LENGTH):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc

# Function to randomly sample frames
def random_sample_frames(mfcc, num_samples=10000, frame_size=N_MELS):
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

# Load and process dataset
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

# Cluster using DBSCAN to estimate the number of clusters
def find_clusters_with_dbscan(data):
    dbscan = DBSCAN(eps=2.5, min_samples=10)
    dbscan.fit(data)
    labels = dbscan.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return num_clusters, labels

# KMeans clustering
def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=300, n_init=10)
    kmeans.fit(data)
    return kmeans.cluster_centers_, kmeans.labels_

# Transform MFCC into Mel Spectrum
def mfcc_to_spectrum(mfcc, sr=sample_rate, n_fft=1024, hop_length=512):
    mel_spectrogram = mfcc_to_mel(mfcc, sr=sr, n_fft=n_fft, hop_length=hop_length)
    return mel_spectrogram

# Function to plot 2D clustering
def plot_2d_clusters(data, labels, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.xlabel('MFCC Coefficient 1')
    plt.ylabel('MFCC Coefficient 2')
    plt.colorbar()
    plt.savefig(f'{title}.png')
    plt.show()

# Go

# Step 1: Convert dataset to MFCC and sample randomly
mfcc_data = process_dataset(dataset_folder, num_samples=10000)

# Step 2: Find number of clusters using DBSCAN
num_clusters, dbscan_labels = find_clusters_with_dbscan(mfcc_data)
print(f'Estimated number of clusters: {num_clusters}')

# Optional: Plot DBSCAN Clusters
# plot_2d_clusters(mfcc_data, dbscan_labels, 'DBSCAN Clustering')

# Step 3: Cluster using KMeans
centroids, kmeans_labels = kmeans_clustering(mfcc_data, num_clusters)
print(f'Cluster centroids:\n{centroids}')

# Optional: Plot KMeans Clusters
# plot_2d_clusters(mfcc_data, kmeans_labels, 'KMeans Clustering')

# Step 4: Centroids to Mel Specturm
mel_spectrograms = []
for centroid in centroids:
    mel_spec = mfcc_to_mel(centroids.T)
    mel_spectrograms.append(mel_spec)

# Convert list to NumPy array
mel_spectrograms = np.array(mel_spectrograms)  # Shape: (num_clusters, n_mels)


# Step 5: Mel Spectrum Filterbank based on the number of clusters, create a mel spectrum filter bank
mel_filterbank = librosa.filters.mel(sr=sample_rate, n_fft=N_FFT, n_mels=num_clusters, fmin=0, fmax=sample_rate / 2)

# Step 6: Get a magnitude of the first mel_spectrograms (there should be num_clusters amount of bands)

# Step 7: Apply magnitude to first band of Mel Spectrum Filterbank based on clusters

# Step 8: Apply a Spectral Shape to Noise



    
    