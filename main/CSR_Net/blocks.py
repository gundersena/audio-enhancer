import tensorflow as tf
from tensorflow.keras import layers, initializers

from CSR_Net.util import SubPixel1D


# ----------------------------------------------------------------------------


class Encoder(layers.Layer):
    '''encodes input audio to higher-dimensions'''

    def __init__(self, n_filters, n_filtersizes, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.w = layers.Conv1D(filters=n_filters, kernel_size=n_filtersizes, padding='same',
            kernel_initializer=initializers.Orthogonal(gain=1.0, seed=None), strides=2)
        self.a = layers.LeakyReLU(0.2)

    def call(self, inputs):
        x = self.w(inputs)
        return self.a(x)


# ----------------------------------------------------------------------------


class Bottleneck(layers.Layer):
    '''middle layer of the network, used to sample data mostly'''

    def __init__(self, n_filters, n_filtersizes, name='bottleneck', **kwargs):
        super(Bottleneck, self).__init__(name=name, **kwargs)
        self.w = layers.Conv1D(filters=n_filters, kernel_size=n_filtersizes, padding='same',
            kernel_initializer=initializers.Orthogonal(gain=1.0, seed=None), strides=2)
        self.n = layers.Dropout(0.5)
        self.a = layers.LeakyReLU(0.2)

    def call(self, inputs):
        x = self.w(inputs)
        x = self.n(x)
        return self.a(x)


# ----------------------------------------------------------------------------



class Decoder(layers.Layer):
    '''decodes (upsamples) high-dimension data back down to audio dimension'''

    def __init__(self, n_filters, n_filtersizes, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.w = layers.Conv1D(filters=2*n_filters, kernel_size=n_filtersizes, padding='same',
            kernel_initializer=initializers.Orthogonal(gain=1.0, seed=None))
        self.n = layers.Dropout(0.5)
        self.a = layers.Activation('relu')
        # (-1, n, f)
        self.s = layers.Lambda(SubPixel1D, arguments={'r':2})

    def call(self, inputs):
        x = self.w(inputs)
        x = self.n(x)
        x = self.a(x)
        return self.s(x)


# ----------------------------------------------------------------------------



class OutputConv(layers.Layer):
    '''output layer for the network'''

    def __init__(self, n_filters, n_filtersizes, name='finalconv', **kwargs):
        super(OutputConv, self).__init__(name=name, **kwargs)
        self.w = layers.Conv1D(filters=2, kernel_size=9, padding='same',
            kernel_initializer=initializers.Orthogonal(gain=1.0, seed=None))
        self.s = layers.Lambda(SubPixel1D, arguments={'r':2})

    def call(self, inputs):
        x = self.w(inputs)
        return self.s(x)
