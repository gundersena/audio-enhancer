# import a dataset and write spectrograms
from matplotlib import pyplot as plt
import numpy as np
import librosa
import h5py


def save_spectrum(S_lr, S_ir, S_pr, outfile, lim=1000):
    plt.subplot(1,3,1)
    plt.title('Data')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.imshow(S_lr, aspect=10)

    plt.subplot(1,3,2)
    plt.title('Label')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.imshow(S_ir, aspect=10)

    plt.subplot(1,3,3)
    plt.title('Label')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.imshow(S_pr, aspect=10)

    plt.tight_layout()
    plt.savefig(outfile)


def get_spectrum(x, n_fft=2048):
    S = librosa.stft(x, n_fft)
    S = np.log1p(np.abs(S))
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S.T


file = 'vctk-train.4.16000.800.10.0.25.h5'

with h5py.File(file, 'r') as hf:
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    print(X)

data = get_spectrum(X.flatten(), n_fft=2048)
label = get_spectrum(Y.flatten(), n_fft=2048)
save_spectrum(data, label, outfile='test1.png')
