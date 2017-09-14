import tensorflow as tf
import tflearn as tfl
from kentf.scoping import adapt_name

def choice(inps, name=None):
    name = adapt_name(name, "choice")
    with tf.variable_scope(name):
        assert len(inps) > 0, "You must provide at least one input."
        shape = [len(inps)] + [int(x) for x in inps[0].shape[1:]]
        W = tf.get_variable("W", shape=shape,
                dtype=tf.float32, initializer=tf.constant_initializer(1.0),
                trainable=True)
        mask = tf.expand_dims(tf.nn.softmax(W, dim=0), 0)
        stacked_inps = tf.stack(inps, axis=1)
        out = tf.reduce_sum(tf.multiply(stacked_inps, mask), axis=1)
        out = tf.identity(out, name)
        print(out)
        return out

def reducechoice(inp, axis, name=None):
    name = adapt_name(name, "reducechoice")
    with tf.name_scope(name):
        avg = tfl.layers.core.flatten(tf.reduce_mean(inp, axis), name="avg")
        maxx = tfl.layers.core.flatten(tf.reduce_max(inp, axis), name="max")
        minn = tfl.layers.core.flatten(tf.reduce_min(inp, axis), name="min")
        pooled = choice([avg, maxx, minn], name=name)
        return pooled
