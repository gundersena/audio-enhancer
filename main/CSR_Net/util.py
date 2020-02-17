import tensorflow as tf
from tensorflow.keras import layers

def SubPixel1D(I, r):
    '''
    One-dimensional subpixel upsampling layer
    Calls a tensorflow function that directly implements this functionality.
    We assume input has dim (batch, width, r)
    '''
    X = tf.transpose(I, [2,1,0]) # (r, w, b)
    X = tf.batch_to_space(X, [r], [[0,0]]) # (1, r*w, b)
    X = tf.transpose(X, [2,1,0])
    return X


# ----------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers


class MergeTensors(layers.Layer):
    """custom layer that handles merging tensors for upscaling purposes"""
    def __init__(self, type, name='merger', **kwargs):
        super(MergeTensors, self).__init__(name=name, **kwargs)
        self.type = type
        self.c = layers.Concatenate(axis=-1)
        self.a = layers.Add()

    def call(self, inputs):
        if self.type == 'concat':
            x = self.c(inputs)
        if self.type == 'add':
            x = self.a(inputs)
        return x
