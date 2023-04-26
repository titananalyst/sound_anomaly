import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def file_names(root, db_levels=None, ids=None, file_types=None, num_files=None):
    files = []
    
    if db_levels is None:
        db_levels = ['0dB', '6dB', 'min6dB']
    
    if ids is None:
        ids = ['id_00', 'id_02', 'id_04', 'id_06']
        
    if file_types is None:
        file_types = ['normal', 'abnormal']

    combinations = len(db_levels) * len(ids) * len(file_types)
    
    if num_files is not None:
        files_per_combination = num_files // combinations
    else:
        files_per_combination = None

    for db_level in db_levels:
        for id_ in ids:
            for file_type in file_types:
                folder_path = os.path.join(root, db_level, 'pump', id_, file_type)
                
                for idx, file_name in enumerate(os.listdir(folder_path)):
                    if files_per_combination is not None and idx >= files_per_combination:
                        break
                    
                    file_path = os.path.join(folder_path, file_name)
                    files.append(file_path)

    return files


def dataloader(files_list, n_fft=1024, hop_length=512, n_mels=64, frames=5):
    dims = n_mels * frames
    
    for idx in tqdm(range(len(files_list)), desc='Dataloader: '):
        signal, sr = file_load(files_list[idx])
        features = extract_features(
        signal,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        frames=frames,
        )
        
        if idx == 0:
            dataset = np.zeros((features.shape[0] * len(files_list), dims), np.float32)
        
        dataset[
            features.shape[0] * idx : features.shape[0] * (idx + 1), :
        ] = features
        
    return dataset
        

def file_load(wav_name, mono=False, channel=0):
    signal, sr = librosa.load(wav_name, mono=False, sr=None)
    if signal.ndim <= 1:
        sound_file = signal, sr
    else:
        sound_file = signal[channel, :], sr

    return sound_file
  
    
def extract_features(signal, sr, n_fft=1024, hop_length=512, 
                     n_mels=64, frames=5):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
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


root = 'Z:\\BA\\mimii_baseline\\dataset'

# Load all files
all_files = file_names(root)

# Load files from specific dB levels and IDs
specific_files = file_names(root, db_levels=['0dB', '6dB'], ids=['id_00', 'id_04'])

# Load normal files only
normal_files = file_names(root, file_types=['normal'])

# Load a limited number of files from each folder
limited_files = file_names(root, num_files=10)

# Load evenly distributed files across specified dB levels, IDs, and file types
evenly_distributed_files = file_names(root, db_levels=['0dB', '6dB'], ids=['id_00', 'id_04'], num_files=20)


six_00_norm = file_names(root, db_levels=['6dB'], ids=['id_00'], file_types=['normal'])
six_00_abnorm = file_names(root, db_levels=['6dB'], ids=['id_00'], file_types=['abnormal'])

six_02_norm = file_names(root, db_levels=['6dB'], ids=['id_02'], file_types=['normal'])
six_02_abnorm = file_names(root, db_levels=['6dB'], ids=['id_02'], file_types=['abnormal'])

six_04_norm = file_names(root, db_levels=['6dB'], ids=['id_04'], file_types=['normal'])
six_04_abnorm = file_names(root, db_levels=['6dB'], ids=['id_04'], file_types=['abnormal'])

six_06_norm = file_names(root, db_levels=['6dB'], ids=['id_06'], file_types=['normal'])
six_06_abnorm = file_names(root, db_levels=['6dB'], ids=['id_06'], file_types=['abnormal'])

six_norm = file_names(root, db_levels=['6dB'], ids=['id_00', 'id_02', 'id_04', 'id_06'], file_types=['normal'])
six_abnorm = file_names(root, db_levels=['6dB'], ids=['id_00', 'id_02', 'id_04', 'id_06'], file_types=['abnormal'])

six_all = file_names(root, db_levels=['6dB'], ids=['id_00', 'id_02', 'id_04', 'id_06'])

print(len(six_00_norm), len(six_00_abnorm))
print(len(six_02_norm), len(six_02_abnorm))
print(len(six_04_norm), len(six_04_abnorm))
print(len(six_06_norm), len(six_06_abnorm))
print(len(six_norm), len(six_abnorm))
print(len(six_all))



# takes about 12-14 minutes
n_fft = 1024
hop_length = 512
n_mels = 64
frames = 5

dim_1 = 313 - frames + 1
dim_2 = n_mels * frames

# Load file loacations
six_norm = file_names(root, db_levels=['6dB'], ids=['id_00', 'id_02', 'id_04', 'id_06'], file_types=['normal'], num_files=50)
six_abnorm = file_names(root, db_levels=['6dB'], ids=['id_00', 'id_02', 'id_04', 'id_06'], file_types=['abnormal'], num_files=50)

# Feature labelling
six_norm_labels = np.zeros(len(six_norm))
six_abnorm_labels = np.ones(len(six_abnorm))


six_norm_data = dataloader(
    six_norm, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, frames=frames)
six_abnorm_data = dataloader(
    six_abnorm, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, frames=frames)


def reshape_data(normal_data, abnormal_data, frames, n_mels):
    dim_1 = 313 - frames + 1
    dim_2 = n_mels * frames

    normal_data_resh = normal_data.reshape(int(normal_data.shape[0] / dim_1), dim_1, dim_2)
    abnormal_data_resh = abnormal_data.reshape(int(abnormal_data.shape[0] / dim_1), dim_1, dim_2)

    return normal_data_resh, abnormal_data_resh


n_mels = 64
frames = 5
six_norm_resh, six_abnorm_resh = reshape_data(six_norm_data, six_abnorm_data, frames=frames, n_mels=n_mels)


def plot_examples(normal_data_resh, abnormal_data_resh, normal_files, n_examples=3):
    fig, axs = plt.subplots(2, n_examples, figsize=(24, 10))
    
    # load one file to get sampling rate
    _, sr = file_load(normal_files[0])
    
    for i in range(n_examples):
        # Plot normal examples
        librosa.display.specshow(
            normal_data_resh[i],
            sr=sr,
            hop_length=hop_length,
            cmap="viridis",
            ax=axs[0, i]
        )
        axs[0, i].set_title(f"Normal Example {i+1}")

        # Plot abnormal examples
        librosa.display.specshow(
            abnormal_data_resh[i],
            sr=sr,
            hop_length=hop_length,
            cmap="viridis",
            ax=axs[1, i]
        )
        axs[1, i].set_title(f"Abnormal Example {i+1}")

    plt.tight_layout()
    plt.show()

plot_examples(six_norm_resh, six_abnorm_resh, six_norm)

# print(six_norm_data[0])
# print(six_norm_resh[0])