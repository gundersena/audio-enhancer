import tensorflow as tf
import numpy as np

from CSR_Net import util, blocks


class MlRes(tf.keras.Model):
    '''the model: piecing together the blocks and defining the logic'''

    def __init__(self):
        super(MlRes, self).__init__()

        # utils
        self.merge1 = util.MergeTensors(type='concat')
        self.merge2 = util.MergeTensors(type='add')

        # n_kernels = [  64,  128,  256, 384, 384, 384, 384, 384]
        # n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
        # n_filters = [  256,  512,  512, 512, 512, 1024, 1024, 1024]
        # n_filtersizes = [129, 65,   33,  17,  9,  9,  9, 9]
        # n_filtersizes = [31, 31,   31,  31,  31,  31,  31, 31]
        # kernel_size = [65, 33, 17,  9,  9,  9,  9, 9, 9]

        # blocks
        self.encode1 = blocks.Encoder(128, 65) # (num_filters, filter_size)
        self.encode2 = blocks.Encoder(256, 33)
        self.encode3 = blocks.Encoder(512, 17)
        # self.encode4 = blocks.Encoder(512,  9)

        self.bottleneck = blocks.Bottleneck(512, 9)

        # self.decode4 = blocks.Decoder(512,  9)
        self.decode3 = blocks.Decoder(512, 17)
        self.decode2 = blocks.Decoder(256, 33)
        self.decode1 = blocks.Decoder(128, 65)

        self.finalconv = blocks.OutputConv(2, 9)

    def call(self, inputs):
        skip = []

        x = self.encode1(inputs)
        skip.append(x)

        x = self.encode2(x)
        skip.append(x)

        x = self.encode3(x)
        skip.append(x)

        # x = self.encode4(x)
        # skip.append(x)


        x = self.bottleneck(x)


        # x = self.decode4(x)
        # x = self.merge1([x, skip[-1]])

        x = self.decode3(x)
        x = self.merge1([x, skip[-1]])

        x = self.decode2(x)
        x = self.merge1([x, skip[-2]])

        x = self.decode1(x)
        x = self.merge1([x, skip[-3]])


        x = self.finalconv(x)
        x = self.merge2([x, inputs])


        return x
