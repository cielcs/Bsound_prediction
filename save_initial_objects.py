import os
import librosa
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

np.set_printoptions(threshold=np.inf)

def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, '%m_%d_%H%M')

def calculate_minutes_difference(datetime1, datetime2):
    delta = datetime2 - datetime1
    return int(delta.total_seconds() / 60)

def extract_mfcc_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    mfcc_std = np.std(mfccs.T, axis=0)
    return np.concatenate((mfcc_mean, mfcc_std))

def process_directory(directory, specified_times):
    mfcc_features_list = []
    labels = []
    colors = []
    file_infos = []

    specified_datetimes = [parse_datetime(time_str) for time_str in specified_times]
    max_minutes_difference = 0

    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            base_name = os.path.splitext(filename)[0]
            file_datetime = parse_datetime(base_name)

            future_datetimes = [dt for dt in specified_datetimes if dt > file_datetime]
            if future_datetimes:
                closest_datetime = min(future_datetimes, key=lambda dt: abs(dt - file_datetime))
                minutes_difference = calculate_minutes_difference(file_datetime, closest_datetime)
                label = f" ({minutes_difference}min.)"
                max_minutes_difference = max(max_minutes_difference, minutes_difference)
            else:
                minutes_difference = None
                label = f"(None)"

            mfccs = extract_mfcc_features(file_path)
            mfcc_features_list.append(mfccs)
            labels.append(label)
            file_infos.append((label, minutes_difference))

    norm = mcolors.Normalize(vmin=0, vmax=max_minutes_difference)
    cmap = plt.colormaps['coolwarm']

    for label, minutes_difference in file_infos:
        if minutes_difference is None:
            color = 'black'
        else:
            color = cmap(norm(minutes_difference))
        colors.append(color)

    return np.array(mfcc_features_list), labels, colors, norm, cmap, file_infos

directory = 'Bsound'
specified_times = ['5_29_1710', '5_29_2235','5_30_1337','5_30_1359','5_31_2144','6_1_1806','6_2_1505',
'6_4_1500','6_5_1842','6_7_1602','6_21_1905','6_23_0132','7_3_1833','7_4_1049','7_5_0306','7_10_1859','7_15_2219','8_11_1258','8_12_1510','9_5_0336','9_7_0125']
mfcc_features, labels, colors, norm, cmap, file_infos = process_directory(directory, specified_times)

# データの標準化
scaler = StandardScaler()
mfcc_features_normalized = scaler.fit_transform(mfcc_features)

# PCAを実行
pca = PCA(n_components=2)
mfcc_pca_normalized = pca.fit_transform(mfcc_features_normalized)

# グループ化と分布の計算
interval = 60  # 分
grouped_data = {}
group_distributions = {}

for i, (label, minutes_difference) in enumerate(file_infos):
    if minutes_difference is not None:
        group_key = (minutes_difference // interval) * interval + interval // 2
        if group_key not in grouped_data:
            grouped_data[group_key] = []
        grouped_data[group_key].append(mfcc_pca_normalized[i])

for group_key, data_points in grouped_data.items():
    if len(data_points) > 1:
        data_points = np.array(data_points)
        mean_group = np.mean(data_points, axis=0)
        cov_group = np.cov(data_points, rowvar=False)
        
        group_distributions[group_key] = {
            'mean': mean_group,
            'cov': cov_group,
            'distribution': multivariate_normal(mean=mean_group, cov=cov_group)
        }

os.makedirs('saved_objects', exist_ok=True)
# オブジェクトを保存
with open('saved_objects/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('saved_objects/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

with open('saved_objects/group_distributions.pkl', 'wb') as f:
    pickle.dump(group_distributions, f)

# plt.show()
