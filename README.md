# Dataset Analysis for pd
 
# Audio MFCC Clustering

Unsupervised clustering of audio using MFCC features, with partial spectral reconstruction.

## What it does

* Extracts MFCCs from `.wav` files
* Removes low-energy frames
* Samples and standardizes data
* Estimates clusters (HDBSCAN)
* Applies K-Means
* Converts cluster centroids → Mel → STFT

## Requirements

```bash
pip install numpy librosa matplotlib scikit-learn soundfile
```

## Usage

Set dataset path in the script, then run:

```bash
python FBE0.3.py
```

## Notes

* Fixed sample rate (44.1 kHz)
* Reconstruction is approximate
* Clustering depends on parameter tuning
