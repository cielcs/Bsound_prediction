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

# 新しい音声ファイルのパス
new_audio_path = 'Bsound/9_7_0058.wav'

# オブジェクトをロード
with open('saved_objects/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('saved_objects/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('saved_objects/group_distributions.pkl', 'rb') as f:
    group_distributions = pickle.load(f)

sorted_group_distributions = dict(sorted(group_distributions.items(), key=lambda item: item[0]))

# 新しい音声ファイルからMFCCを抽出し、標準化とPCAを適用
new_mfcc = extract_mfcc_features(new_audio_path)
new_mfcc_normalized = scaler.transform([new_mfcc])
new_mfcc_pca = pca.transform(new_mfcc_normalized)
new_mfcc_pca = new_mfcc_pca.flatten()

# 各グループの分布に対して尤度を計算
# likelihoods = {}
# for group_key, distribution in sorted_group_distributions.items():
#     likelihood = distribution['distribution'].pdf(new_mfcc_pca)
#     likelihoods[group_key] = likelihood

#古典的に正規分布をそれぞれ仮定して、その分布に対する尤度を計算
likelihoods2 = {}
first_prior = {}
Marginal_likelihood = 0
Posterior = {}
for group_key, distribution in sorted_group_distributions.items():
    mean = distribution['mean']
    cov = distribution['cov']
    likelihood2 = np.exp(-0.5 * (new_mfcc_pca - mean).T @ np.linalg.inv(cov) @ (new_mfcc_pca - mean)) / ((2*np.pi)**(len(mean)/2)*np.sqrt(np.linalg.det(cov)))
    likelihoods2[group_key] = likelihood2
    first_prior[group_key] = 1/len(sorted_group_distributions)
    Marginal_likelihood += likelihood2 * first_prior[group_key]

for(group_key, distribution) in sorted_group_distributions.items():
    Posterior[group_key] = likelihoods2[group_key] * first_prior[group_key] / Marginal_likelihood

# print("Likelihoods:", likelihoods)
print("Likelihoods2:", likelihoods2)
print("First prior:", first_prior)
print("Marginal likelihood:", Marginal_likelihood)
print("Posterior:", Posterior)
# print(sorted_group_distributions.keys())


# データの準備
keys = list(likelihoods2.keys())
likelihoods = [likelihoods2[key] for key in keys]
priors = [first_prior[key] for key in keys]
posteriors = [Posterior[key] for key in keys]

# グラフの作成
plt.figure(figsize=(10, 6))

plt.plot(keys, likelihoods, label='Likelihoods', marker='o')
plt.plot(keys, priors, label='Priors', marker='o')
plt.plot(keys, posteriors, label='Posteriors', marker='o')

plt.xlabel('Key')
plt.ylabel('Value')
plt.title('Likelihoods, Priors, and Posteriors by Key')
plt.legend()

plt.grid(True)
plt.show()

