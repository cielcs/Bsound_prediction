import os
import librosa
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

np.set_printoptions(threshold=np.inf)

def parse_datetime(datetime_str):
    # 指定された形式の文字列を日時オブジェクトに変換します
    return datetime.strptime(datetime_str, '%m_%d_%H%M')

def calculate_minutes_difference(datetime1, datetime2):
    # 2つの日時の差分を分単位で計算します
    delta = datetime2 - datetime1
    return int(delta.total_seconds() / 60)

def extract_mfcc_features(audio_path, n_mfcc=13):
    # 音声ファイルからMFCC特徴量を抽出します。
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    mfcc_std = np.std(mfccs.T, axis=0)
    return np.concatenate((mfcc_mean, mfcc_std))

def process_directory(directory, specified_times):
    # 指定されたディレクトリ内の音声ファイルを処理し、MFCC特徴量、ラベル、色などを返します。
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
            # print(filename)
            # print(future_datetimes) # 測定時刻より後の指定時刻のリスト
            if future_datetimes:
                closest_datetime = min(future_datetimes, key=lambda dt: abs(dt - file_datetime))
                # print(f"Closest datetime: {closest_datetime}") # 測定時刻より後の指定時刻のうち、最も近い時刻
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

# ディレクトリのパス
directory = '/Users/shota/programs/m1/research/Bsound'
specified_times = ['5_29_1710', '5_29_2235','5_30_1337','5_30_1359','5_31_2144','6_1_1806','6_2_1505',
'6_4_1500','6_5_1842','6_7_1602','6_21_1905','6_23_0132','7_3_1833','7_4_1049','7_5_0306']
# MFCCを抽出
mfcc_features, labels, colors, norm, cmap, file_infos = process_directory(directory, specified_times)

# データの標準化
scaler = StandardScaler()
mfcc_features_normalized = scaler.fit_transform(mfcc_features)
# print(mfcc_features_normalized)

# PCAを実行
pca = PCA(n_components=2)
mfcc_pca_normalized = pca.fit_transform(mfcc_features_normalized)

# PCAを標準化しないMFCCに対して実行
pca_non_standardized = PCA(n_components=2)
mfcc_pca_non_standardized = pca_non_standardized.fit_transform(mfcc_features)

# プロット
fig, axs = plt.subplots(1, 2, figsize=(20, 7))

# 標準化したデータのプロット
axs[0].scatter(mfcc_pca_normalized[:, 0], mfcc_pca_normalized[:, 1], c=colors)
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')
axs[0].set_title('PCA of Standardized MFCC Features')
axs[0].grid(True)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[0], label='Minutes Difference')

# 標準化しないデータのプロット
axs[1].scatter(mfcc_pca_non_standardized[:, 0], mfcc_pca_non_standardized[:, 1], c=colors)
axs[1].set_xlabel('Principal Component 1')
axs[1].set_ylabel('Principal Component 2')
axs[1].set_title('PCA of Non-standardized MFCC Features')
axs[1].grid(True)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[1], label='Minutes Difference')


# グループ化と分布の計算
interval = 60  # 分
grouped_data = {}
group_distributions = {}  # 各グループの正規分布を格納する辞書

for i, (label, minutes_difference) in enumerate(file_infos):
    if minutes_difference is not None:
        group_key = (minutes_difference // interval) * interval + interval // 2
        if group_key not in grouped_data:
            grouped_data[group_key] = []
        grouped_data[group_key].append(mfcc_pca_normalized[i])

# 楕円をプロットする際に、特定のサブプロットに対して add_patch を呼び出す必要があります。
for group_key, data_points in grouped_data.items():
    if len(data_points) > 1:
        data_points = np.array(data_points)
        mean_group = np.mean(data_points, axis=0)
        cov_group = np.cov(data_points, rowvar=False)
        
        # 正規分布のパラメータを格納
        group_distributions[group_key] = {
            'mean': mean_group,
            'cov': cov_group,
            'distribution': multivariate_normal(mean=mean_group, cov=cov_group)
        }
        
        # 共分散行列から楕円の主軸の長さと角度を計算
        eigvals, eigvecs = np.linalg.eigh(cov_group)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

        # 標準偏差1倍の範囲の楕円をプロット
        ell_radius_x = np.sqrt(eigvals[0])
        ell_radius_y = np.sqrt(eigvals[1])
        ellipse = Ellipse(
            mean_group,
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            angle=angle,
            edgecolor=cmap(norm(group_key)),
            fc='None',
            lw=2,
            label=f'{group_key} min group'
        )
        # ここで、特定のサブプロットに楕円を追加します。
        axs[0].add_patch(ellipse)  # axs[1] にも同じように楕円を追加したい場合は、axs[1] を使用します。

plt.show()
