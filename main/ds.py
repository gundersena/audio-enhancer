import os, argparse
import numpy as np
import h5py
import random

import librosa
from scipy import interpolate
from scipy.signal import decimate

from scipy.signal import butter, lfilter


class Prep_VCTK:
    """main class for creating data pipelines"""
    # we just changed the add_data function into the __init__ method
    def __init__(self, type, num_files, dim, file_list,
                 scale=4,
                 interpolate=True,
                 low_pass=False,
                 batch_size=32,
                 sr=16000,
                 sam=0.25):
        self.type = type
        self.num_files = num_files
        self.scale = scale
        self.dim = dim
        self.stride = dim
        self.interpolate = interpolate
        self.low_pass = low_pass
        self.batch_size = batch_size
        self.sr = sr
        self.sam = sam

        self.path = '../data/multispeaker'
        out = f'{self.path}/vctk-{self.type}.{self.scale}.{self.sr}.{self.dim}.{self.num_files}.{self.sam}.h5'
        with h5py.File(out, 'w') as f: # create h5 file for data to be placed in
            self.add_data(h5_file=f, inputfiles=file_list, save_examples=False)

    def add_data(self, h5_file, inputfiles, save_examples=False):
        # Make a list of all files to be processed
        file_list = []
        file_extensions = set(['.wav'])
        with open(inputfiles) as f:
            for line in f: # for every file in the .txt file
                filename = line.strip() # strips any spaces off of the filename
                ext = os.path.splitext(filename)[1]
                if ext in file_extensions: # if file is wavefile, add to file_list
                    file_list.append(filename) # add path of filename before adding
        file_list = random.sample(file_list, int(self.num_files))

      # patches to extract and their size
        if self.interpolate: # if user wants to replace low-res patches with cubpic splines
            d, d_lr = self.dim, self.dim # dimensions for lr and sd are the same
            s, s_lr = self.stride, self.stride # extracting low-res stride
        else: # apply scaling to lr audio
            d, d_lr = self.dim, self.dim / self.scale
            s, s_lr = self.stride, self.stride / self.scale
        hr_patches, lr_patches = list(), list()

        for j, file_path in enumerate(file_list): #update user on progress (ie. 30/240)
            if j % 10 == 0:
                print (f'Making {self.type} data...{int(np.ceil(j/self.num_files*100))}%    \r', end='')
            # load audio file from file_list
            x, fs = librosa.load(f'../data/{file_path}', sr=self.sr)

            # crop so that it works with scaling ratio (ie. divisible by 2, 4, 6, etc.)
            x_len = len(x) # length of file
            x = x[ : x_len - (x_len % self.scale)]

            # generate low-res version
            if self.low_pass:
                # x_bp = butter_bandpass_filter(x, 0, args.sr / args.scale / 2, fs, order=6)
                # x_lr = np.array(x[0::args.scale])
                #x_lr = decimate(x, args.scale, zero_phase=True)
                x_lr = decimate(x, self.scale) # downsample signal after applying anti-aliasing filter
            else:
                x_lr = np.array(x[0::self.scale]) #just sample audio at a lower rate (every 2, 4, 6, etc.)

            if self.interpolate: # zero padd array to have same dim as HD (ie, 4000300020001)
                x_lr = Prep_VCTK.upsample(self, x_lr)
                assert len(x) % self.scale == 0
                assert len(x_lr) == len(x)
            else:
                assert len(x) % self.scale == 0
                assert len(x_lr) == len(x) / self.scale

            # generate patches
            max_i = len(x) - int(d) + 1 # max iteration?: file length - dimension + 1
            # iterate through the file in strides
            for i in range(0, max_i, s):
                # keep only a fraction of all the patches (not in use)
                u = np.random.uniform() # a single value is returned between 0 and 1
                if u > self.sam: continue # only keeping a random % of the patches if args.sam is specified

            if self.interpolate:
                i_lr = i
            else:
                i_lr = i / self.scale

            hr_patch = np.array( x[i : i+d] ) # current patch = current position + dim
            lr_patch = np.array( x_lr[i_lr : i_lr+d_lr] )

            # print 'a', hr_patch
            # print 'b', lr_patch

            assert len(hr_patch) == d
            assert len(lr_patch) == d_lr

            # print hr_patch

            hr_patches.append(hr_patch.reshape((d,1))) # create hr patches
            lr_patches.append(lr_patch.reshape((d_lr,1))) # create lr patches

            # if j == 1: exit(1)

        # crop # of patches so that it's a multiple of mini-batch size
        num_patches = len(hr_patches)
        print (f'num_patches = {num_patches}')
        num_to_keep = int(np.floor(num_patches / self.batch_size) * self.batch_size)
        hr_patches = np.array(hr_patches[:num_to_keep])
        lr_patches = np.array(lr_patches[:num_to_keep])

        print (hr_patches.shape)

        # create the hdf5 file
        data_set = h5_file.create_dataset('data', lr_patches.shape, np.float32)
        label_set = h5_file.create_dataset('label', hr_patches.shape, np.float32)

        # fill hdf5 files with patches
        data_set[...] = lr_patches
        label_set[...] = hr_patches

    def upsample(self, x_lr): #lr = lowres, hr = highres
        x_lr = x_lr.flatten() # flatten audio array
        x_hr_len = len(x_lr) * self.scale # get (len of audio array * scaling factor)
        x_sp = np.zeros(x_hr_len) # create zero-padded array with new length

        i_lr = np.arange(x_hr_len, step=self.scale) # create lr array with step size of scaling factor
        i_hr = np.arange(x_hr_len)

        f = interpolate.splrep(i_lr, x_lr) # "Given the set of data points (x[i], y[i]) determine a smooth spline approximation"

        # Given the knots and coefficients of a B-spline representation, evaluate the value of the smoothing polynomial and its derivatives.
        x_sp = interpolate.splev(i_hr, f)

        return x_sp

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
