#!/usr/bin/env python

import pickle
import os
import sys

import numpy as np
import matplotlib.pylab as plt

import yaml
import json

import scipy.stats as stats
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import metrics
# import scikitplot as skplt
from tqdm import tqdm

import librosa

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


def load_all_models(directory, base_dir):
    models = []
    model_info = []

    for dir_idx, target_dir in enumerate(base_dir):
        print("\n[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(base_dir)))
        # Extract dataset parameters

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

####################################################################################

# Visualizer
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# color reference: "RdYlGn" (orange is darkened here)
color = ["#d7191c", "#d7191c", "#ff952d", "#ff952d","#1a9641", "#1a9641"]
vector_list = [0, 0.2999999, 0.3, 0.6999999, 0.7, 1]
color_list = list(zip(vector_list, color))
cmap = LinearSegmentedColormap.from_list('rg', color_list, N=256)

class Visualizer(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 17))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def grid_plot(self, data, model_id, metric):
        """plot"""
        # Preprocess data
        df_list = []
        for model, model_data in data.items():
            for eval_db, eval_db_data in model_data.items():
                for model_db, metrics in eval_db_data.items():
                    df_list.append([f"{model}_{eval_db}", model_db, metrics[metric]])

        df = pd.DataFrame(df_list, columns=["Model", "Model_DB", metric])
        # print(df)

        # Pivot the data to make it suitable for a heatmap
        pivot_df = df.pivot_table(values=metric, index='Model', columns='Model_DB')
        # print(pivot_df)
        
        # Reindex to ensure order
        pivot_df = pivot_df.reindex(index=[f'id_{model_id}_min6dB', f'id_{model_id}_0dB', f'id_{model_id}_6dB'], columns=['min6dB', '0dB', '6dB'])
        # print(pivot_df)
        
        # Draw a heatmap
        ax = self.fig.add_subplot(1,1,1)
        ax.cla()
        heatmap = sns.heatmap(pivot_df, annot=True, cmap=cmap, vmin=0, vmax=1,
                              linewidths=2,
                              linecolor="white", 
                              annot_kws={"size":18}, 
                              cbar_kws={'label': 'Score %',
                                        "ticks":[0, 0.3, 0.5, 0.7, 1]},)
        
        ax.set_title(f"Model Performance ({metric})", fontsize=22, pad=24)
        ax.set_xlabel('testset_dB', fontsize=22)
        ax.set_ylabel(f'model id_{model_id}', fontsize=22)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # rotation labels x,y axis
        ax.tick_params(axis='both', labelsize=18)
        cax = ax.figure.axes[-1]  # access colorbar
        cax.tick_params(labelsize=18) # cbar tick size
        cax.yaxis.label.set_size(20) # label "Accuracy %"


    def recon_plot(self, data, label, n_abnorm, perc, eval_dB):
        """
        input:  data:   y_pred as reconstruction error of all files
                label:  machine type, id and db
                n_abnorm: number of abnormal files for the train eval split
        """
        self.fig = plt.figure(figsize=(30, 10))
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.scatter(np.arange(len(data[:n_abnorm])), data[:n_abnorm], label='normal')
        ax.scatter(np.arange(len(data[n_abnorm:])) + n_abnorm, data[n_abnorm:], label='abnormal')
        ax.axhline(y=1, color='black', linestyle='-', label=f'Percentile ({perc*100} %)')
        ax.set_title(f'Anomaly Score - Model {label} on testset {eval_dB}', fontsize=22, pad=24)
        ax.set_xlabel('File Index', fontsize=22)
        ax.set_ylabel('Anomaly Score', fontsize=22)
        ax.legend(loc="upper left", fontsize=22)
        ax.tick_params(axis='both', labelsize=22)
        
        # define y limit, scatter dots are better
        # visible if not starting at 0
        lb = min(data) - 0.05 if min(data) < 0.6 else 0.55
        ub = max(data) + 0.05 if max(data) > 2.0 else 2.05
        ax.set_ylim(lb, ub)

    def conf_matrix_plot(self, y_true, y_pred, label, eval_dB):
        self.fig = plt.figure(figsize=(15, 15))
        
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
        # print(conf_matrix)
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size=22)
        
        ax.set_title(f'Confusion Matrix - Model {label} on testset {eval_dB}', fontsize=22, pad=24)
        ax.set_xlabel('Predictions', fontsize=22)
        ax.set_ylabel('Actuals', fontsize=22)

        ax.xaxis.set_major_locator(plt.FixedLocator([0, 1]))  # Set x-axis major locator
        ax.xaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))  # Set x-axis formatter

        ax.yaxis.set_major_locator(plt.FixedLocator([0, 1]))  # Set y-axis major locator
        ax.yaxis.set_major_formatter(plt.FixedFormatter(['Negative', 'Positive']))  # Set y-axis formatter
        ax.tick_params(axis='both', labelsize=18)

    def roc_curve_plot(self, y_true, y_pred, label, eval_dB):
        # precision recall curve
        ax = self.fig.add_subplot(1,1,1)
        ax.cla()
        fpr, tpr, _= metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(y_true, y_pred)
        # roc = metrics.RocCurveDisplay.from_predictions(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        roc = metrics.RocCurveDisplay.from_predictions(y_true, y_pred, label=label)

        roc.plot(ax=ax)
        ax.set_xlabel('False Positive Rate', fontsize=22)
        ax.set_ylabel('True Positive Rate', fontsize=22)
        ax.set_title(f'ROC Curve - Model {label} on testset {eval_dB}', fontsize=22, pad=24)
        ax.legend(fontsize=22, loc="lower right")
        ax.tick_params(axis='both', labelsize=22)

    def save_figure(self, name):
        self.fig.savefig(name + ".pdf", dpi=300)
        print('Image saved!')
        self.fig.clf()

####################################################################################
if __name__ == '__main__':
    evaluation = True
    anomaly_plot = True
    conf_plot = True
    roc_plot = True
    gen_plot = True


    # load parameter yaml
    with open("test.yaml") as stream:
        param = yaml.safe_load(stream)
    
    # Load base directory list
    # dirs = sorted(glob.glob(os.path.abspath("{base}/*/*/*".format(base=param["base_directory"]))))

    # pump id 
    id = '06'
    dirs = [f'Z:\\BA\\mimii_baseline\\dataset\\6dB\\pump\\id_{id}', f'Z:\\BA\\mimii_baseline\\dataset\\0dB\\pump\\id_{id}', f'Z:\\BA\\mimii_baseline\\dataset\\min6dB\\pump\\id_{id}']
    print(dirs)

    results = {}
    # filepath for results
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f'{date_str}_grid_result.json'
    results_dir = param['result_directory']
    filepath = os.path.join(results_dir, filename)

    ##############################################################
    # model evaluation
    if evaluation == True:
        # dir for anomaly score plots
        grid_eval_dir = param['result_model_anomaly']
        grid_eval_path = f'{grid_eval_dir}/{date_str}'

        # dir for conf_matrix plots
        conf_matrix_dir = param['result_conf_matrix']
        conf_matrix_path = f'{conf_matrix_dir}/{date_str}'

        # dir for roc curve
        roc_curve_dir = param['result_roc_curve']
        roc_curve_path = f'{roc_curve_dir}/{date_str}'

        if not os.path.exists(grid_eval_dir):
            os.makedirs(grid_eval_dir)

        if not os.path.exists(grid_eval_path):
            os.makedirs(grid_eval_path)

        if not os.path.exists(conf_matrix_dir):
            os.makedirs(conf_matrix_dir)

        if not os.path.exists(conf_matrix_path):
            os.makedirs(conf_matrix_path)   

        if not os.path.exists(roc_curve_dir):
            os.makedirs(roc_curve_dir)

        if not os.path.exists(roc_curve_path):
            os.makedirs(roc_curve_path)       

        # load models
        model_dir = param["model_directory"]
        models, model_info = load_all_models(model_dir, dirs)
        print("Loaded all models from", model_dir)
        # print(models)
        # print(model_info)

        all_data = load_data_from_directory(param, dirs)
        # print("")
        # print(all_data)
    
        db_levels = ['6dB', '0dB', 'min6dB']

        for model, info in zip(models, model_info):
            print('\nevaluation')
            print(info)
            # Extract machine_type, machine_id, db from info
            machine_type = info["machine_type"]
            machine_id = info["machine_id"]
            model_db = info["db"]

            # Create a dictionary for this machine_id if it doesn't exist yet
            if machine_id not in results:
                results[machine_id] = {}

            # Create a dictionary for this model dB level if it doesn't exist yet
            if model_db not in results[machine_id]:
                results[machine_id][model_db] = {}
            
            # dot_img_file = "img/model_1.pdf"
            # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
            # print("Model plotted!")
            # model.summary()
            # load model
            autoencoder = model
                
            # Iterate over db levels
            for eval_db in db_levels:
                print(f"\nEvaluating model {info['model_name']} with eval_data: {eval_db}")

                # Check if data exists for this db level
                if eval_db in all_data[machine_type][machine_id]:
                    print(f'Load eval data for {machine_type}_{machine_id}_{eval_db}')
                    # Extract the data for this model and db level
                    eval_data = all_data[machine_type][machine_id][eval_db]

                    # Set the variables
                    eval_files = eval_data['eval_files']
                    eval_labels = eval_data['eval_labels']
                    n_norm_abnorm = eval_data['n_norm_abnorm']
                    min_val, max_val = eval_data['norm_values']

                    evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                            machine_id=machine_id,
                                                                            db=model_db)
                    
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
                            # mse
                            error = np.mean(np.square(data - autoencoder.predict(data, verbose=0)), axis=1)
                            # mae
                            # error = np.mean(abs(data - autoencoder.predict(data, verbose=0)), axis=1)
                            y_pred[num] = np.mean(error)
                            
                        except:
                            print('File broken:', file)

                    # calculate mean and std of the normal data
                    normal_error = [y_pred[i] for i in range(len(y_pred)) if y_true[i] == 0]
                    mean, std = np.mean(normal_error), np.std(normal_error)

                    # percentile of the normal distribution
                    percentile = 0.998  # 99.8 percentile
                    threshold = stats.norm.ppf(percentile, loc=mean, scale=std)
                    anomaly_score = [score / threshold for score in y_pred]
                    y_pred = anomaly_score
                    print(y_pred)

                    # binary prediction based on the anomaly score
                    y_pred_binary = [1 if score > 1 else 0 for score in anomaly_score]
                    print(y_pred_binary)

                    # print('error', error)
                    # print(np.shape(error))
                    # print('y_pred', y_pred)
                    # print(np.shape(y_pred))

                    # AUC
                    auc_score = metrics.roc_auc_score(y_true, y_pred)
                    print("AUC : {}".format(auc_score))                    
                    
                    # F1
                    # threshold = np.median(y_pred)
                    # y_pred_binary = [1 if pred > threshold else 0 for pred in y_pred]
                    f1_score = metrics.f1_score(y_true, y_pred_binary)
                    print("F1 Score : {}".format(f1_score))
                    
                    # precision, recall
                    precision = metrics.precision_score(y_true, y_pred_binary)
                    recall = metrics.recall_score(y_true, y_pred_binary)
                    print("Precision : {}".format(precision))
                    print("Recall : {}".format(recall))

                    # roc curve
                    
                    results[machine_id][model_db][eval_db] = {
                        "AUC": auc_score,
                        "F1": f1_score,
                        "Precision": precision,
                        "Recall": recall
                    }


                    # plot anomaly score
                    if anomaly_plot == True:
                        vis_recon = Visualizer()
                        vis_recon.recon_plot(y_pred, evaluation_result_key, n_norm_abnorm[1], percentile, eval_db)
                        vis_recon.save_figure(f'{grid_eval_path}/model_id_{id}_{model_db}_testset_{eval_db}_anomaly_score')

                    if conf_plot == True:
                        vis_conf = Visualizer()
                        vis_conf.conf_matrix_plot(y_true, y_pred_binary, evaluation_result_key, eval_db)
                        vis_conf.save_figure(f'{conf_matrix_path}/model_id_{id}_{model_db}_testset_{eval_db}_conf_matrix')

                    if roc_plot == True:
                        vis_roc = Visualizer()
                        vis_roc.roc_curve_plot(y_true, y_pred_binary, evaluation_result_key, eval_db)
                        vis_roc.save_figure(f'{roc_curve_path}/model_id_{id}_{model_db}_testset_{eval_db}_roc_curve')

                else:
                    print(f"Warning: No data found for db level {eval_db} for model {info['model_name']}")

        print(results)
        with open(filepath, 'w') as f:
            json.dump(results, f)
            print('Results saved to file: {}'.format(filepath))


    if gen_plot == True:
        with open(filepath, 'r') as f:
            results = json.load(f)
        print(results)
        
        # create directories for saving the grid results
        grid_eval_dir = param['result_grid_directory']
        grid_eval_path = f'{grid_eval_dir}/{date_str}'

        if not os.path.exists(grid_eval_dir):
            os.makedirs(grid_eval_dir)

        if not os.path.exists(grid_eval_path):
            os.makedirs(grid_eval_path)

        # visualize grids
        visualiser = Visualizer()
        print(f'{grid_eval_path}/model_id_{id}_AUC')
        visualiser.grid_plot(results, id, "AUC")
        visualiser.save_figure(f'{grid_eval_path}/model_id_{id}_AUC')

        visualiser.grid_plot(results, id, "F1")
        visualiser.save_figure(f'{grid_eval_path}/model_id_{id}_F1')

        visualiser.grid_plot(results, id, "Precision")
        visualiser.save_figure(f'{grid_eval_path}/model_id_{id}_Precision')

        visualiser.grid_plot(results, id, "Recall")
        visualiser.save_figure(f'{grid_eval_path}/model_id_{id}_Recall')