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

"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

####################################################################################

# Directories
PICKLE_DIR = 'D:\9999_OneDrive_ZHAW\OneDrive - ZHAW\BA_ZHAW_RTO\pickle'
PICKLE_DIR_SMALL = 'D:\9999_OneDrive_ZHAW\OneDrive - ZHAW\BA_ZHAW_RTO\pickle\subset'
root = 'Z:\\BA\\mimii_baseline\\dataset'

####################################################################################

# Visualizer
import matplotlib.pyplot as plt

class Visualizer(object):
    def __init__(self):
        self.fig = plt.figure(figsize=(30, 10), dpi=1000)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss, label):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss, label="Train")
        ax.plot(val_loss, label="Test")
        ax.set_title(f'Model loss - {label}', fontsize=22)
        ax.set_xlabel("Epoch", fontsize=22)
        ax.set_ylabel("Loss", fontsize=22)
        ax.legend(loc="upper right", fontsize=22)
        ax.tick_params(axis='both', labelsize=22)

    def recon_plot(self, data, label, n_abnorm, perc):
        """
        input:  data:   y_pred as reconstruction error of all files
                label:  machine type, id and db
                n_abnorm: number of abnormal files for the train eval split
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.scatter(np.arange(len(data[:n_abnorm])), data[:n_abnorm], label='normal')
        ax.scatter(np.arange(len(data[n_abnorm:])) + n_abnorm, data[n_abnorm:], label='abnormal')
        ax.axhline(y=1, color='black', linestyle='-', label=f'Percentile ({perc})')
        ax.set_title(f'Anomaly Detection - {label}', fontsize=22)
        ax.set_xlabel('File Index', fontsize=22)
        ax.set_ylabel('Anomaly Score', fontsize=22)
        ax.legend(loc="upper left", fontsize=22)
        ax.tick_params(axis='both', labelsize=22)

    def pr_curve_plot(self, true, pred, label):
        # precision recall curve
        ax = self.fig.add_subplot(1,1,1)
        ax.cla()
        precision, recall, _ = precision_recall_curve(true, pred)
        # print(np.shape(precision), np.shape(recall))
        # print(precision, recall)
        prd = PrecisionRecallDisplay(precision, recall)
        prd.plot(ax=ax)
        ax.set_xlabel('Recall', fontsize=22)
        ax.set_ylabel('Precision', fontsize=22)
        ax.set_title(f'Precision Recall Curve - {label}', fontsize=22)
        ax.tick_params(axis='both', labelsize=22)
        
    def save_figure(self, name):
        """
        Save figure.

        name : str
            save .png file path.

        return : None
        """
        self.fig.savefig(name)
        print('Image saved!')
        self.fig.clf()


####################################################################################

def dataloader(files_list, n_fft=1024, hop_length=512, n_mels=64, frames=5, pwr=2, msg='Dataloader: '):
    dims = n_mels * frames
    
    for idx in tqdm(range(len(files_list)), desc=msg):
        signal, sr = file_load(files_list[idx])
        features = extract_features(
        signal,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        frames=frames,
        power=pwr
        )
        
        if idx == 0:
            dataset = np.zeros((features.shape[0] * len(files_list), dims), np.float32)
        
        dataset[
            features.shape[0] * idx : features.shape[0] * (idx + 1), :
        ] = features
        
    return dataset
        
def file_dataloader(file_name, n_fft=1024, hop_length=512, n_mels=64, frames=5, pwr=2):
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



def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """
    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    normal_labels = np.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    abnormal_labels = np.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("train_labels num : {num}".format(num=len(train_labels)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))
    logger.info("eval_labels  num : {num}".format(num=len(eval_labels)))

    len_norm = len(normal_files)
    len_abnorm = len(abnormal_files)

    return train_files, train_labels, eval_files, eval_labels, len_norm, len_abnorm

####################################################################################
# evaluation

def best_model_ae(log_loss, log_label, path):
    """
    input:  log_loss_val
            log_label_2
            path: MODEL_PATH_2
    """
    # Select model with best loss on validation set!!
    log_loss = np.array(log_loss)
    mask = np.ravel(log_loss==min(log_loss))
    label = list(itertools.compress(log_label, mask))[0]
    print('Loaded Model: ', label)

    model = tf.keras.models.load_model(path + 'model_AE_' + str(label) + '.h5')
    # X_hat_train = model.predict(x=X_train)
    # X_hat_val = model.predict(x=X_val)
    # X_hat_test = model.predict(x=X_test)
    return model

####################################################################################
# Model
####################################################################################

def create_encoder(input_shape, config):
    """
    Creates an encoder network with an architecture
    following a geometric series where each hidden layer 
    has half the number of neurons as the previous layer
    inputs:
        input_shape: tuple with input shape
        config: dictionary with nn configuration
    outputs:
        z: np.array with lantent space
        encoder: tf model        
    """ 
    
    # Architecture
    latent_dim = config['n_ls_a']  # latent space dim
    cells = [int(config['n_cl_a']*(0.5)**i) for i in range(config['n_hl_a'])]
    
    
    # Define the inputs, input layer
    X_inputs = tf.keras.Input(shape=input_shape, name='encoder_input') 
    X = X_inputs
    
    # Encoding, going through the hidden layers (cells defined above)
    for i in range(config['n_hl_a']):
        X = tf.keras.layers.Dense(cells[i],
                                  activation=config['activ'],
                                  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=config['seed']))(X)
        
    # Latent vector
    z = tf.keras.layers.Dense(latent_dim, name='z')(X)  
    
    # Build encoder from 'X_inputs' to 'z' space
    encoder = tf.keras.Model(X_inputs, z, name='encoder')
    
    return z, encoder

def create_decoder(output_shape, config):
    """
    inputs:
        input_shape: tuple with input shape
        config: dictionary with nn configuration
    outputs:
        outputs: np.array with reconstruction signal
        decoder: tf model        
    """
    
    # Architecture
    latent_dim = config['n_ls_a']
    cells = [int(config['n_cl_a']*(0.5)**i) for i in range(config['n_hl_a'])]
    
    # Define the inputs (from vector to time-dependent input)
    Z_inputs = tf.keras.Input(shape=(latent_dim,))
    X = Z_inputs
    
    # Dencoding      
    for i in reversed(range(config['n_hl_a'])):
        X = tf.keras.layers.Dense(cells[i],                                       
                                  activation=config['activ'],
                                  kernel_initializer=tf.keras.initializers.glorot_uniform(seed=config['seed']))(X)

    # Reconstructed input
    outputs = tf.keras.layers.Dense(output_shape[-1])(X)
    
    # Build decoder model
    decoder = tf.keras.Model(Z_inputs, outputs, name='decoder')
    # rebuilds the input to the output
    
    return outputs, decoder

def create_autoencoder(input_shape, output_shape, config):
    """
    inputs:
        input_shape: tuple with input shape
        output_shape: tuple with output shape (in case it is an asymetric AE)
        config: dictionary with nn configuration
    outputs:
        autoencoder, encoder, decoder: tf models        
    """
    
    # Define the inputs
    X_inputs = tf.keras.Input(shape=input_shape) 
    
    # Create encoder
    z, encoder = create_encoder(input_shape, config) 
    
    # Create decoder
    outputs, decoder = create_decoder(output_shape, config)
   
    # Create autoencoder
    X_hat = decoder(encoder(X_inputs))
    autoencoder = tf.keras.Model(X_inputs, X_hat, name='ae')    

    # Optimiser set-up
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
    
    # Compilation
    autoencoder.compile(optimizer=opt, loss="mean_squared_error")
    
    return autoencoder, encoder, decoder


def predict():
    pass


def fit_model_ul(OUTFOLDER, 
                   X_train, Y_train, 
                   X_val, Y_val,
                   X_test, Y_test,
                   config, label, generate=True):
    """
    Creates and trains a NN with unsupervised learning strategy: 
    define model shapes, create model, fit model, plot training loss and save model
    inputs:
        OUTFOLDER: path to storage or model folder
        X_{train, val, test}: np.array with train, val and test input features.
        Y_{train, val, test}: np.array with train, val and test target features.
        config: dictionary with NN configuration (i.e. hyperparameters).
        label: str with model name for storage or loading
        generate_a: boolean with load or run.
    outputs:
        loss_val: np.array with loss in val dataset
        Y_hat_{train, val, test}: np.array with train, val and test output predictions.       
    """
    if generate:
        # Set-up I - 
        seed = config['seed']
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed)
        tf.keras.backend.clear_session()        
        
        # Set-up II - Shapes 
        input_shape = X_train.shape[1:]
        output_shape = X_train.shape[1:]
        
        # Create autoencoder model
        autoencoder, encoder, decoder = create_autoencoder(input_shape, output_shape, config)
        
        # Callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], 
                                                      restore_best_weights = True)
        
        # Report model summary
        autoencoder.summary()
        # Report model summary
        encoder.summary()
        
        # Fit model       
        history = autoencoder.fit(X_train, X_train,
                                  batch_size=config['batch_size'], 
                                  epochs=config['epochs'], 
                                  callbacks=[early_stop],
                                  #validation_data=(X_val, Y_val),
                                  validation_split=0.1,
                                  verbose=1)   
        
        # Summarize history for loss
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'model loss - {label}', fontsize=22)
        plt.ylabel('loss', fontsize=22)
        plt.xlabel('epoch', fontsize=22)
        plt.legend(['train', 'val'], loc='upper left', fontsize=22)
        plt.tick_params(axis='both', labelsize=22)
        plt.savefig('D:/9999_OneDrive_ZHAW/OneDrive - ZHAW/BA_ZHAW_RTO/img/loss/' + str(label) + '_' + strftime("%Y-%m-%d", gmtime()) +'.png')
        plt.clf()
        # plt.show()

        # Save encoder model
        encoder.save(OUTFOLDER + 'model_E_' + str(label) + '.h5')
        print('')
        print("Saved Encoder model to disk")
        
        # Save auteoencoder model
        autoencoder.save(OUTFOLDER + 'model_AE_' + str(label) + '.h5')
        print('')
        print("Saved AutoEncoder model to disk")
        
        # Save decoder model
        decoder.save(OUTFOLDER + 'model_D_' + str(label) + '.h5')
        print('')
        print("Saved Decoder model to disk")     
        
    else:
        # Autoencoder
        autoencoder= tf.keras.models.load_model(OUTFOLDER + 'model_AE_' + str(label) + '.h5')
        print('')
        print("Loaded Autoencoder model from disk")
        
        # Report model summary
        autoencoder.summary()
        
        # Compilation
        autoencoder.compile(optimizer='Adam', loss="mean_squared_error") 
        
    # Evaluate model      
    loss_val = autoencoder.evaluate(x=X_val, y=X_val)
    
    # Predict outputs
    Y_hat_train = autoencoder.predict(x=X_train)
    Y_hat_val = autoencoder.predict(x=X_val)
    Y_hat_test = autoencoder.predict(x=X_test)

    return loss_val, Y_hat_train, Y_hat_val, Y_hat_test


def grid_search_ul(MODEL_PATH,
                   X_train, Y_train, 
                   X_val, Y_val,
                   X_test, Y_test, 
                   params, n_runs, varInput, generate=True):
    """
    Performs a grid search in a NN with unsupervised learning strategy: 
    define model shapes, create model, fit model, plot training loss and save model
    inputs:
        MODEL_PATH: path to storage or model folder.
        X_{train, val, test}: np.array with train, val and test input features.
        params: dictionary, NN possible configurations.
        n_runs: int, each configuration is performed n_runs times.
        varInput: str with mapping label.
        generate_a: boolean with load or run.
    outputs:
        df: storage dataframe.
        log_label: list of unique model labels.
    """
    # Set-up
    df = pd.DataFrame()
    log_loss_val, log_label, log_df = [], [], []
    keys, values = zip(*params.items())

    for kk, bundle in enumerate(product(*values)):        # Varing architectures
        
        # Architecture [kk]
        config = dict(zip(keys, bundle))

        for jj in range(n_runs):                         # Check reproducibility - n runs             
            df_k = pd.DataFrame(config, index=[0])
           
            # Define simulation label
            label =  varInput + '_h_a_' + str(kk) + '_run_' + str(jj)
            print('')
            print('Simulation:', label)

            # Fit NN model
            time_start = time.time()
            loss_val, Y_hat_train, Y_hat_val, Y_hat_test =\
            fit_model_ul(MODEL_PATH,
                           X_train, Y_train, 
                           X_val, Y_val,
                           X_test, Y_test,
                           config, label, generate=generate)
            
            # Store results
            log_loss_val.append(loss_val)
            log_label.append(label)

            # Log architecture/run/results as pandas DataFrame
            df_k['run']= jj
            df_k['RMSE-Ts'] = np.round(np.sqrt(np.mean((Y_hat_test - Y_test)**2)), 3)
            df_k['RMSE-Va'] = np.round(np.sqrt(np.mean((Y_hat_val - Y_val)**2)), 3)
            df_k['RMSE-Tr'] = np.round(np.sqrt(np.mean((Y_hat_train - Y_train)**2)), 3)
            df_k['Time[min]'] = np.round((time.time()-time_start)/60, 2)            
            log_df.append(df_k)
            df = pd.concat(log_df, ignore_index=True)

            print('')
            print(df.to_string())

            # Write solutions to 
            df.to_csv(MODEL_PATH + 'Training_US_' +  varInput + '.csv')

    return df, log_label, log_loss_val


####################################################################################
if __name__ == '__main__':
    one_machine = False
    result_generation = True
    normalization = True
    gen_grid_plot = True
    
    # main 
    # load parameter yaml
    with open("test.yaml") as stream:
        param = yaml.safe_load(stream)

    # laod base directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/*/*/*".format(base=param["base_directory"]))))

    if one_machine == True:
        # dirs = [dirs[0]]
        # dirs = ['Z:\\BA\\mimii_baseline\\dataset\\6dB\\pump\\id_00']
        dirs = ['Z:\\BA\\mimii_baseline\\dataset\\6dB\\pump\\id_00', 'Z:\\BA\\mimii_baseline\\dataset\\6dB\\pump\\id_02', 'Z:\\BA\\mimii_baseline\\dataset\\6dB\\pump\\id_04', 'Z:\\BA\\mimii_baseline\\dataset\\6dB\\pump\\id_06']
        print(dirs)

    # initialize visualizer
    visualizer = Visualizer()

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    for dir_idx, target_dir in enumerate(dirs):
        print("\n[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))
     
        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        eval_pickle = "{pickle}/eval_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        train_files_pickle = "{pickle}/train_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_files_pickle = "{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        train_labels_pickle = "{pickle}/train_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_labels_pickle = "{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        n_norm_abnorm_pickle = "{pickle}/n_norm_abnorm_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        norm_values_pickle = "{pickle}/norm_values_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        model_file = "{model}/model_{machine_type}_{machine_id}_{db}.hdf5".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)
        history_img = "{model}/history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        recon_img = "{model}/recon_error_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        pr_curve_img = "{model}/pr_curve_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)

        # costumize results file
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        evaluation_result['Date_Time'] = dt_string
        evaluation_result['Config'] = param['config']

        # generate dataset
        if os.path.exists(train_pickle) and os.path.exists(train_labels_pickle) and os.path.exists(train_files_pickle) and os.path.exists(eval_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle) and os.path.exists(n_norm_abnorm_pickle) and os.path.exists(norm_values_pickle):
            train_data = load_pickle(train_pickle)
            train_files = load_pickle(train_files_pickle)
            train_labels = load_pickle(train_labels_pickle)
            eval_data = load_pickle(eval_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
            n_norm_abnorm =load_pickle(n_norm_abnorm_pickle)
            min_val, max_val = load_pickle(norm_values_pickle)

        else:        
            print("Generating dataset")
            train_files, train_labels, eval_files, eval_labels, n_norm, n_abnorm = dataset_generator(target_dir)

            train_data = dataloader(train_files,
                                    n_fft = param['feature']['n_fft'],
                                    hop_length = param['feature']['hop_length'],
                                    n_mels = param['feature']['n_mels'],
                                    frames = param['feature']['frames'],
                                    pwr = param['feature']['power'],
                                    msg="Training data: ")
            
            eval_data = dataloader(eval_files,
                        n_fft = param['feature']['n_fft'],
                        hop_length = param['feature']['hop_length'],
                        n_mels = param['feature']['n_mels'],
                        frames = param['feature']['frames'],
                        pwr = param['feature']['power'],
                        msg="Evaluation data: ")
            if normalization == True:
                min_val = np.min(train_data, axis = 0)
                max_val = np.max(train_data, axis = 0)
                print(f'Norm_values generation: min {min_val}, max {max_val}')
                save_pickle(norm_values_pickle, (min_val, max_val))
            
            n_norm_abnorm = (n_norm, n_abnorm)
            
            print(train_data)
            print(np.shape(train_data))
            save_pickle(train_pickle, train_data)
            save_pickle(train_files_pickle, train_files)
            save_pickle(train_labels_pickle, train_labels)
            save_pickle(eval_pickle, eval_data)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)
            save_pickle(n_norm_abnorm_pickle, n_norm_abnorm)

        print(train_data)
        print(eval_data)
        # normalization of train and eval dataset
        min_val, max_val = load_pickle(norm_values_pickle)
        train_data, _, _ = normalize_data(train_data, min_val, max_val)
        eval_data, _, _ = normalize_data(eval_data, min_val, max_val)
        print(train_data)
        print(eval_data)

       
        ###############################################################
        # normal approach without grid_search

        # model training
        print("model training")
        # create autoencoder
        autoencoder, encoder, decoder = create_autoencoder(train_data.shape[1:], 
                                                           train_data.shape[1:], 
                                                           param['config'])
        
        # Callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=param['config']['patience'], 
                                                      restore_best_weights = True)

        # report model summary
        autoencoder.summary()
        # report model summary
        encoder.summary()

        if os.path.exists(model_file):
            autoencoder = tf.keras.models.load_model(model_file)
        else:
            history = autoencoder.fit(train_data, train_data,
                                    batch_size = param['config']['batch_size'],
                                    epochs = param['config']['epochs'],
                                    callbacks = [early_stop],
                                    # validation_data = (eval_data, eval_data)
                                    validation_split = 0.1)
            
            visualizer.loss_plot(history.history['loss'], history.history['val_loss'], evaluation_result_key)
            visualizer.save_figure(history_img)
            autoencoder.save(model_file)


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
                
                if normalization == True:
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

    if result_generation == True:
        with open(result_file, "w") as f:
                f.write(yaml.dump(results, default_flow_style=False))
                print("Results saved!")


    if gen_grid_plot == True:
        pass
