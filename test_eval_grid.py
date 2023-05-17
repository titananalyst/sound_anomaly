#!/usr/bin/env python

import pickle
import os
import sys
import glob

import numpy as np
import matplotlib.pylab as plt

import time

import scipy.stats as stats
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

##################################################################################
# eval grid

# def load_all_models(directory):
#     models = []
#     model_info = []

#     # Get all files in the directory
#     all_files = os.listdir(directory)

#     # Filter for only the .hdf5 files
#     hdf5_files = [f for f in all_files if f.endswith('.hdf5')]

#     # Load each model
#     for model_file in hdf5_files:
#         model_path = os.path.join(directory, model_file)
#         model = tf.keras.models.load_model(model_path)

#         # Extract machine_type, machine_id, db from the filename
#         parts = model_file.replace('model_', '').replace('.hdf5', '').split('_')
#         print(parts)
#         db = parts[-1]
#         machine_type =  parts[0]# Join remaining parts as machine_type
#         machine_id = '_'.join(parts[-3:-1]) 

#         # Append the loaded model and the extracted info to the lists
#         models.append(model)
#         model_info.append({
#             "model_name": model_file,
#             "machine_type": machine_type,
#             "machine_id": machine_id,
#             "db": db
#         })

#     return models, model_info

def load_all_models(directory, base_dir):
    models = []
    model_info = []

    for dir_idx, target_dir in enumerate(base_dir):
        print("\n[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(base_dir)))
        # Extract dataset parameters
        # db, machine_type, machine_id = os.path.normpath(target_dir).split(os.sep)[-3:]

        # Define the corresponding model filename

        # Extract dataset parameters 
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]

        model_file = f"model_{machine_type}_{machine_id}_{db}.hdf5"

        model_path = os.path.join(directory, model_file)
        print("loaded model", model_path)

        # Check if the model file exists before trying to load it
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)

            # Append the loaded model and the extracted info to the lists
            models.append(model)
            model_info.append({
                "model_name": model_file,
                "machine_type": machine_type,
                "machine_id": machine_id,
                "db": db
            })
        else:
            print(f"Warning: Expected model file does not exist: {model_path}")

    return models, model_info


def load_data_from_directory(param, base_dir):
    # Initialize a dictionary to hold all the data
    all_data = {}

    for dir_idx, target_dir in enumerate(base_dir):
        print("\n[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(base_dir)))
        
        # Extract dataset parameters 
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]

        # Setup paths
        file_types = ["train", "eval", "train_files", "eval_files", "train_labels", "eval_labels", "n_norm_abnorm", "norm_values"]
        for file_type in file_types:
            file_path = "{pickle}/{file_type}_{machine_type}_{machine_id}_{db}.pickle".format(
                pickle=param["pickle_directory"],
                file_type=file_type,
                machine_type=machine_type,
                machine_id=machine_id,
                db=db
            )

            # Load the pickle file
            data = load_pickle(file_path)

            # Store the data in the dictionary, creating sub-dictionaries as necessary
            if machine_type not in all_data:
                all_data[machine_type] = {}
            if machine_id not in all_data[machine_type]:
                all_data[machine_type][machine_id] = {}
            if db not in all_data[machine_type][machine_id]:
                all_data[machine_type][machine_id][db] = {}

            all_data[machine_type][machine_id][db][file_type] = data

    return all_data


####################################################################################
if __name__ == '__main__':

    # load parameter yaml
    with open("test.yaml") as stream:
        param = yaml.safe_load(stream)
    
        # Load base directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/*/*/*".format(base=param["base_directory"]))))
    dirs = ['Z:\\BA\\mimii_baseline\\dataset\\6dB\\pump\\id_00', 'Z:\\BA\\mimii_baseline\\dataset\\0dB\\pump\\id_00', 'Z:\\BA\\mimii_baseline\\dataset\\min6dB\\pump\\id_00']
    
    # load models
    model_dir = param["model_directory"]
    models, model_info = load_all_models(model_dir, dirs)
    print("Loaded all models from", model_dir)
    # print(models)
    # print(model_info)

    all_data = load_data_from_directory(param, dirs)
    # print("")
    # print(all_data)


    ##############################################################
    # model evaluation
    db_levels = ['min6dB', '0dB', '6dB']

    for model, info in zip(models, model_info):
        print('evaluation')
        print(info, '\n')
        # Extract machine_type, machine_id, db from info
        machine_type = info["machine_type"]
        machine_id = info["machine_id"]
        
        # Iterate over db levels
        for db in db_levels:
            print(f"Evaluating model {info['model_name']} with eval_data: {db}")

            # Check if data exists for this db level
            if db in all_data[machine_type][machine_id]:
                print(f'Load data for {machine_type}_{machine_id}_{db}')
                # Extract the data for this model and db level
                eval_data = all_data[machine_type][machine_id][db]

                # Set the variables
                eval_files = eval_data['eval_files']
                eval_labels = eval_data['eval_labels']
                n_norm_abnorm = eval_data['n_norm_abnorm']
                min_val, max_val = eval_data['norm_values']

                # model evaluation
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
                        
                        
                        data, _, _ = normalize_data(data, min_val, max_val)
                        # error = np.mean(np.square(data - autoencoder.predict(data)), axis=1)
                        error = np.mean(abs(data - autoencoder.predict(data, verbose=0)), axis=1)
                        y_pred[num] = np.mean(error)
                        
                    except:
                        print('File broken:', file)

                # threshold of the normal data and the anomaly score by relative proportion
                # percentile = 99.8
                # threshold = np.percentile([y_pred[i] for i in range(len(y_pred)) if y_true[i] == 0], percentile)
                # anomaly_score = [score / threshold for score in y_pred]
                # y_pred = anomaly_score

                # # binary prediction based on the anomaly score
                # y_pred_binary = [1 if score > 1 else 0 for score in anomaly_score]


                # calculate mean and std of the normal data
                normal_error = [y_pred[i] for i in range(len(y_pred)) if y_true[i] == 0]
                mean, std = np.mean(normal_error), np.std(normal_error)

                # percentile of the normal distribution
                percentile = 0.998  # 99.8 percentile
                threshold = stats.norm.ppf(percentile, loc=mean, scale=std)
                anomaly_score = [score / threshold for score in y_pred]
                y_pred = anomaly_score

                # binary prediction based on the anomaly score
                y_pred_binary = [1 if score > 1 else 0 for score in anomaly_score]

                print('error', error)
                print(np.shape(error))
                print('y_pred', y_pred)
                print(np.shape(y_pred))

                # threshold with kmeans clustering
                # y_pred_array = np.array(y_pred)
                # y_pred_resh = y_pred_array.reshape(-1, 1)
                # kmeans = KMeans(n_clusters=2, random_state=0).fit(y_pred_resh)
                # centroids = kmeans.cluster_centers_
                # threshold = np.mean(centroids)

                # AUC
                score = metrics.roc_auc_score(y_true, y_pred)
                print("AUC : {}".format(score))
                evaluation_result["AUC"] = float(score)
                

                # F1
                # threshold = np.median(y_pred)
                # y_pred_binary = [1 if pred > threshold else 0 for pred in y_pred]
                f1_score = metrics.f1_score(y_true, y_pred_binary)
                print("F1 Score : {}".format(f1_score))
                evaluation_result["F1"] = float(f1_score)

                results[evaluation_result_key] = evaluation_result

                visualizer.recon_plot(y_pred, evaluation_result_key, n_norm_abnorm[1], percentile)
                visualizer.save_figure(recon_img)

                # precision recall curve
                visualizer.pr_curve_plot(y_true, y_pred, evaluation_result_key)
                visualizer.save_figure(pr_curve_img)


            else:
                print(f"Warning: No data found for db level {db} for model {info['model_name']}")

        
        