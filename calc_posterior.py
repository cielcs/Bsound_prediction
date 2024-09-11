import librosa
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt

def extract_mfcc_features(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    mfcc_std = np.std(mfccs.T, axis=0)
    return np.concatenate((mfcc_mean, mfcc_std))

# オブジェクトをロード
with open('saved_objects/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('saved_objects/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('saved_objects/group_distributions.pkl', 'rb') as f:
    group_distributions = pickle.load(f)

sorted_group_distributions = dict(sorted(group_distributions.items(), key=lambda item: item[0]))

# 初期事前分布を一様分布で設定
prior = {key: 1/len(sorted_group_distributions) for key in sorted_group_distributions}

def update_posterior(audio_path, prior, sorted_group_distributions):
    new_mfcc = extract_mfcc_features(audio_path)
    new_mfcc_normalized = scaler.transform([new_mfcc])
    new_mfcc_pca = pca.transform(new_mfcc_normalized)
    new_mfcc_pca = new_mfcc_pca.flatten()

    likelihoods = {}
    marginal_likelihood = 0
    posterior = {}

    for group_key, distribution in sorted_group_distributions.items():
        mean = distribution['mean']
        cov = distribution['cov']
        likelihood = np.exp(-0.5 * (new_mfcc_pca - mean).T @ np.linalg.inv(cov) @ (new_mfcc_pca - mean)) / \
                     ((2*np.pi)**(len(mean)/2)*np.sqrt(np.linalg.det(cov)))
        likelihoods[group_key] = likelihood
        marginal_likelihood += likelihood * prior[group_key]

    for group_key in sorted_group_distributions:
        posterior[group_key] = likelihoods[group_key] * prior[group_key] / marginal_likelihood

    return posterior, likelihoods, marginal_likelihood

# 複数の音声データを観測してベイズ更新
audio_paths = ['Bsound/9_5_0135.wav', 'Bsound/9_5_0235.wav', 'Bsound/9_5_0324.wav', 'Bsound/9_7_0122.wav', 'Bsound/9_7_0123.wav']  # 観測する音声データをリスト化




# グラフ全体の大きさを設定（例えば、横に4つずつ配置）
num_steps = len(audio_paths)
num_cols = 3  # 一列に何個のグラフを並べるか
num_rows = (num_steps + num_cols - 1) // num_cols  # グラフの行数

fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*4))  # 各プロットのサイズ調整

axes = axes.flatten()  

for i, audio_path in enumerate(audio_paths):
    posterior, likelihoods, marginal_likelihood = update_posterior(audio_path, prior, sorted_group_distributions)

    # グラフを各ステップごとに描画
    keys = list(likelihoods.keys())
    likelihoods_values = [likelihoods[key] for key in keys]
    priors_values = [prior[key] for key in keys]
    posteriors_values = [posterior[key] for key in keys]

    axes[i].plot(keys, likelihoods_values, label='Likelihoods', marker='o')
    axes[i].plot(keys, priors_values, label='Priors', marker='o')
    axes[i].plot(keys, posteriors_values, label='Posteriors', marker='o')

    axes[i].set_xlabel('Probability')
    axes[i].set_ylabel('Value')
    axes[i].set_title(f'After {audio_path}')
    axes[i].legend()
    axes[i].grid(True)

    prior = posterior 

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout() 
plt.show()

