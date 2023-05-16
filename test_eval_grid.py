#!/usr/bin/env python

import pickle
import os
import sys
import glob

import numpy as np
import matplotlib.pylab as plt

import time

import itertools
import pandas as pd
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from sklearn import metrics
from sklearn.cluster import KMeans
from pandas import DataFrame
from itertools import product
from matplotlib import gridspec
from operator import itemgetter
from tqdm import tqdm
from time import localtime, gmtime, strftime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

import yaml
import logging
import librosa


####################################################################################

####################################################################################

# Directories
PICKLE_DIR = 'D:\9999_OneDrive_ZHAW\OneDrive - ZHAW\BA_ZHAW_RTO\pickle'
PICKLE_DIR_SMALL = 'D:\9999_OneDrive_ZHAW\OneDrive - ZHAW\BA_ZHAW_RTO\pickle\subset'
root = 'Z:\\BA\\mimii_baseline\\dataset'

####################################################################################

def file_dataloader(file_name, n_fft=1024, hop_length=512, n_mels=64, frames=5, pwr=2, msg='Dataloader: '):
    """
    Function for loading and extracting features form single files.
    """
    signal, sr = file_load(file_name)
    features = extract_features(
    signal,
    sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    frames=frames,
    power=pwr)
        
    return features


def file_load(wav_name, mono=False, channel=0):
    signal, sr = librosa.load(wav_name, mono=mono, sr=None)
    if signal.ndim <= 1:
        sound_file = signal, sr
    else:
        sound_file = signal[channel, :], sr

    return sound_file
  
    

def extract_features(signal, sr, n_fft=1024, hop_length=512, 
                     n_mels=64, frames=5, power = 2): # added power
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power # added power
    )
    # here was power to db
    # added log scale
    # print('mel_spectrogram:\n', mel_spectrogram)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    # print('\nlog_mel_spectrogram:\n', log_mel_spectrogram)
    features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
    
    dims = frames * n_mels

    if features_vector_size < 1:
        return np.empty((0, dims), np.float32)

    features = np.zeros((features_vector_size, dims), np.float32)
    for time in range(frames):
        features[:, n_mels * time : n_mels * (time + 1)] = log_mel_spectrogram[
            :, time : time + features_vector_size
        ].T

    return features


# pickle I/O
def save_pickle(filename, save_data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data


# Normalization
def normalize_data(x, lb, ub, max_v=1.0, min_v=-1.0):
    '''
    Max-Min normalize of 'x' with max value 'max_v' min value 'min_v'
    '''

    # Set-up
    if len(ub)==0:
        ub = x.max(0) # OPTION 1
        # applied to the first dimension (0) columns of the data
        #ub = np.percentile(x, 99.9, axis=0, keepdims=True) # OPTION 2:
        
    if len(lb)==0:
        lb = x.min(0) 
        #lb = np.percentile(x, 0.1, axis=0, keepdims=True)
    
    ub.shape = (1,-1)
    lb.shape = (1,-1)           
    max_min = max_v - min_v
    delta = ub-lb

    # Compute
    x_n = max_min * (x - lb) / delta + min_v
    if 0 in delta:
        idx = np.ravel(delta == 0)
        x_n[:,idx] = x[:,idx] - lb[:, idx]

    return x_n, lb, ub 


####################################################################################
if __name__ == '__main__':

    # Define SNR levels and initialize result matrices
    snrs = ['min6dB', '0dB', '6dB']
    auc_results = np.zeros((len(snrs), len(snrs)))
    f1_results = np.zeros((len(snrs), len(snrs)))

    # Loop over all combinations of training and testing SNRs
    for i, train_snr in enumerate(snrs):
        for j, test_snr in enumerate(snrs):
            # Load training data
            train_data, train_labels = load_pickle(f"dataset/{train_snr}dB/train.pkl")

            # Initialize and train model
            model = YourModel()  # Replace with your model class
            model.fit(train_data, train_labels)

            # Load testing data
            eval_files, eval_labels = load_pickle(f"dataset/{test_snr}dB/test.pkl")

            # Model evaluation
            print('evaluation')
            y_pred = [0. for k in eval_labels]
            y_true = eval_labels

            for num, file in tqdm(enumerate(eval_files), total = len(eval_files)):
                try:
                    data = file_dataloader(file,
                                    n_fft = param['feature']['n_fft'],
                                    hop_length = param['feature']['hop_length'],
                                    n_mels = param['feature']['n_mels'],
                                    frames = param['feature']['frames'],
                                    pwr = param['feature']['power'])
                    
                    if normalization == True:
                        data, _, _ = normalize_data(data, min_val, max_val)

                    error = np.mean(np.square(data - autoencoder.predict(data)), axis=1)
                    y_pred[num] = np.mean(error)
                    
                except:
                    print('File broken)')

            # compute the 99.9 percentile of the normal data
            threshold = np.percentile([y_pred[i] for i in range(len(y_pred)) if y_true[i] == 0], 99.9)

            # compute the anomaly score
            anomaly_score = [score / threshold for score in y_pred]

            # replace y_pred with the anomaly score
            y_pred = anomaly_score

            # create binary predictions: 1 if anomaly_score > 1 else 0
            y_pred_binary = [1 if score > 1 else 0 for score in anomaly_score]

            # AUC
            score = roc_auc_score(y_true, y_pred)
            print("AUC : {}".format(score))
            auc_results[i, j] = float(score)
            

            # F1
            f1_score_res = f1_score
            f1_score_res = f1_score(y_true, y_pred_binary)
            print("F1 Score : {}".format(f1_score_res))
            f1_results[i, j] = float(f1_score_res)
