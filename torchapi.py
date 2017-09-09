import tensorflow as tf
import tflearn as tfl
from kentf.scoping import adapt_name

def pad(tensor, padW, padH, name=None):
    name = adapt_name(name, "pad")
    return tf.pad(tensor, [[0, 0], [padW, padW], [padH, padH], [0, 0]], name=name)

def SpatialConvolution(inp, _, nfilters, kW, kH, dW, dH, padW, padH, **kwargs):
    name = adapt_name(kwargs.get("name", None), "conv")
    with tf.variable_scope(name):
        out = inp
        out = pad(out, padW, padH)
        config = dict(
            strides=(dW, dH),
            padding='valid',
            regularizer='L2',
            weights_init='xavier',
            bias_init='zeros',
            weight_decay=1.0,
        )
        config.update(kwargs)
        out = tfl.layers.conv.conv_2d(out, nfilters, (kW, kH), **config)
    return out

def SpatialMaxPooling(inp, kW, kH, dW=1, dH=1, padW=0, padH=0, **kwargs):
    name = adapt_name(kwargs.get("name", None), "pool")
    with tf.name_scope(name):
        out = inp
        out = pad(out, padW, padH)
        config = dict(
            strides=(dW, dH),
            padding='valid',
        )
        config.update(kwargs)
        out = tfl.layers.conv.max_pool_2d(out, (kW, kH), **config)
    return out
