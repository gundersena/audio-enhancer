from keras.callbacks import Callback
import keras.backend as K
import numpy as np


class SGDRScheduler(Callback):
    """custom callback for implementing a SGDR learning rate"""
    def __init__(self, min_lr, max_lr, steps_per_epoch, lr_decay=0.9, cycle_length=10,
                    mult_factor=1.5):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_lenrfrrgth = cycle_length
        self.mult_factor = mult_factor


    def clr(self):
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr


    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_lr)


    def on_batch_end(self, batch, logs=None):
        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())


    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay


# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback


class LRFinder(Callback):
    """
    custom callback for evaluating the optimal lr range for SGDR
    Usage:
    lr_finder = models.util.LRFinder(min_lr=1e-5, max_lr=3e-2,
        steps_per_epoch=np.ceil(n_sam/args.batch_size), epochs=3)
    """
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super(LRFinder, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}


    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr-self.min_lr) * x


    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)


    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.tight_layout()
        plt.savefig('plots/lr.png')
        plt.clf()

    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig('plots/loss.png')
        plt.clf()


# ----------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import h5py
import ds

def load_data(args, type, num_files, full_data=False):
    np.set_printoptions(threshold=100)
    path = '../data/multispeaker'

    # load training data
    datasets = os.listdir(path)

    for dataset in datasets:
        if str(args.dim_size) and str(num_files) in dataset:
            if args.new_data == 'False':
                make_data = False
                break
        else:
            make_data = True

    if make_data:
        ds.Prep_VCTK(type=type, num_files=num_files, dim=args.dim_size, file_list=f'{path}/{type}-files.txt')

    with h5py.File(f'{path}/vctk-{type}.4.16000.{args.dim_size}.{num_files}.0.25.h5', 'r') as hf:
        X = np.array(hf.get('data'))
        Y = np.array(hf.get('label'))

    n_sam, n_dim, n_chan = Y.shape
    r = Y[0].shape[1] / X[0].shape[1]

    if full_data:
        return X, Y, n_sam
    else:
        return X, Y


# ----------------------------------------------------------------------------
import os


def review_model(args, x_train, y_train):
    """reviews model parameters and raises warnings if something is not recommended"""
    # prints preivew of the data
    preview_data(x_train, y_train)

    # assert not overwriting weights
    if args.from_ckpt == 'False':
        files = os.listdir('./logs/weights')
        for file in files:
            if f'loss.{args.model_id}' in file:
                input('Warning: Are you sure you want to write over these weights?')


# ----------------------------------------------------------------------------
import numpy as np


def preview_data(X, Y):
    print ('Preview X:')
    print (f'Shape: {X.shape}')
    print (f'Max: {np.amax(X)} | Min: {np.amin(X)}')
    print (X[1])

    print ('Preview Y:')
    print (f'Shape of Y: {Y.shape}')
    print (f'Max: {np.amax(Y)} | Min: {np.amin(Y)}')
    print (Y[1])

    # data = eval_wav.get_spectrum(X[:100].flatten(), n_fft=2048)
    # label = eval_wav.get_spectrum(Y[:100].flatten(), n_fft=2048)

    input('Press enter to continue...')


# ----------------------------------------------------------------------------
import os
import librosa
import numpy as np
from keras.models import Model
from scipy import interpolate
from scipy.signal import decimate
from matplotlib import pyplot as plt


class eval_wav:
    '''
    Helper function for eval() in main.py
    Takes a single wavfile and evaluates it by exporting audio and spectrogram
        for hr, lr, and pr
    '''
    def __init__(self, file, args, model):
        # ../data/VCTK-Corpus---
        x_hr, fs = librosa.load(file, sr=args.sample_rate)

        # ensure that input is a multiple of 2^downsampling layers
        ds_layers = 5
        x_hr = eval_wav.clip(x_hr, 2**ds_layers)
        assert len(x_hr) % 2**ds_layers == 0

        # downscale signal
        # x_lr = decimate(x_hr, args.scale)
        x_lr = np.array(x_hr[0::args.scale])
        # x_lr = downsample_bt(x_hr, args.scale)
        assert len(x_hr)/len(x_lr) == args.scale


        # upsample signal through interpolation
        x_ir = eval_wav.upsample(x_lr, args.scale)
        assert len(x_ir) == len(x_hr)

        # trim array again to make it a multiple of 800
        x_ir = eval_wav.clip(x_ir, 800)
        print(f'Input length: {len(x_ir)}')


        n_sam = len(x_ir)/800
        x_pr = model.predict(x_ir.reshape(int(n_sam), 800, 1))
        x_pr = x_pr.flatten()

        # save the file
        filename = os.path.basename(file)
        name = os.path.splitext(filename)[-2]

        if args.make_audio:
            audio_data = np.concatenate((x_hr, x_ir, x_pr), axis=0)
            audio_outname = f'../samples/audio/{name}'
            librosa.output.write_wav(audio_outname + '.hr.wav', audio_data, fs)

        # save the spectrum
        spec_outname = f'../samples/spectrograms/{name}'
        self.outfile=spec_outname + '.png'
        self.S_pr = eval_wav.get_spectrum(x_pr, n_fft=2048)
        self.S_hr = eval_wav.get_spectrum(x_hr, n_fft=2048)
        self.S_lr = eval_wav.get_spectrum(x_lr, n_fft=2048/args.scale)
        self.S_ir = eval_wav.get_spectrum(x_ir, n_fft=2048)
        self.save_spectrum()

    @staticmethod
    def upsample(x_lr, r): #lr = lowres, hr = highres

      x_lr = x_lr.flatten() # flatten audio array
      x_hr_len = len(x_lr) * r # get (len of audio array * scaling factor)
      x_sp = np.zeros(x_hr_len) # create zero-padded array with new length

      i_lr = np.arange(x_hr_len, step=r) # create lr array with step size of scaling factor
      i_hr = np.arange(x_hr_len)

      f = interpolate.splrep(i_lr, x_lr) # "Given the set of data points (x[i], y[i]) determine a smooth spline approximation"

      # Given the knots and coefficients of a B-spline representation, evaluate the value of the smoothing polynomial and its derivatives.
      x_sp = interpolate.splev(i_hr, f)

      return x_sp

    @staticmethod
    def clip(array, multiple):
        x_len = len(array)
        remainder = x_len % multiple
        x_len = x_len - remainder
        array = array[:x_len]
        return array

    @staticmethod
    def get_spectrum(data, n_fft=2048):
        S = librosa.stft(data, int(n_fft))
        S = np.log1p(np.abs(S))
        p = np.angle(S)
        S = np.log1p(np.abs(S))
        return S.T

    def save_spectrum(self, lim=1000):
        plt.subplot(2,2,1)
        plt.title('Target')
        plt.xlabel('Frequency')
        plt.ylabel('Time')
        plt.imshow(self.S_hr, aspect=10)
        plt.xlim([0,lim])

        plt.subplot(2,2,2)
        plt.title('Test')
        plt.xlabel('Frequency')
        plt.ylabel('Time')
        plt.imshow(self.S_lr, aspect=10)
        plt.xlim([0,lim])

        plt.subplot(2,2,3)
        plt.title('Interp')
        plt.xlabel('Frequency')
        plt.ylabel('Time')
        plt.imshow(self.S_ir, aspect=10)
        plt.xlim([0,lim])

        plt.subplot(2,2,4)
        plt.title('Predict')
        plt.xlabel('Frequency')
        plt.ylabel('Time')
        plt.imshow(self.S_pr, aspect=10)
        plt.xlim([0,lim])

        plt.tight_layout()
        plt.savefig(self.outfile)
