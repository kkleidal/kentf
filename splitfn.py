import tensorflow as tf
import tflearn as tfl
from kentf.scoping import adapt_name

# Useful for breaking up very large batch sizes to avoid allocating large tensors:
def splitfn(inp, fn, maxbatch=None, allow_unrolling=True, name=None):
    name = adapt_name(name, "splitfn")
    with tf.variable_scope(name) as scope:
        if not allow_unrolling or inp.shape[0].value is None:
            leng = tf.shape(inp)[0]
            def minibatch():
                scope.reuse_variables()
                remainder = tf.mod(leng, maxbatch, name="remainder")
                splits = tf.identity(tf.floor_div(leng - remainder, maxbatch), "splits")
                remainder_inp = tf.slice(inp,
                        [leng - remainder if i == 0 else 0 for i in range(len(inp.shape))],
                        [-1 for i in range(len(inp.shape))])
                majority_inp = tf.slice(inp,
                        [0 for i in range(len(inp.shape))],
                        [leng - remainder if i == 0 else -1 for i in range(len(inp.shape))])
                split_inp = tf.reshape(
                    majority_inp,
                    tf.concat([[splits, maxbatch], tf.shape(inp)[1:]], 0))
                majority_out = tf.map_fn(fn, split_inp)
                scope.reuse_variables()
                remainder_out = fn(remainder_inp)
                out = tf.concat([
                    tf.reshape(majority_out,
                        tf.concat([[leng - remainder], tf.shape(majority_out)[2:]], 0)),
                    remainder_out], 0)
                if inp.shape[0].value is not None:
                    out = tf.reshape(out, tf.concat([[int(inp.shape[0])], tf.shape(out)[1:]], 0))
                return out
            if maxbatch is None:
                out = fn(inp)
            else:
                out = tf.case([(maxbatch < leng, minibatch)], lambda: fn(inp))
        else:
            leng = int(inp.shape[0])
            if maxbatch is not None and maxbatch < leng:
                remainder = leng % maxbatch
                splits = (leng - remainder) // maxbatch
                remainder_inp = tf.slice(inp,
                        [leng - remainder if i == 0 else 0 for i in range(len(inp.shape))],
                        [-1 for i in range(len(inp.shape))])
                majority_inp = tf.slice(inp,
                        [0 for i in range(len(inp.shape))],
                        [leng - remainder if i == 0 else -1 for i in range(len(inp.shape))])
                split_inp = tf.reshape(
                    majority_inp,
                    tf.concat([[splits, maxbatch], tf.shape(inp)[1:]], 0))
                majority_out = tf.map_fn(fn, split_inp)
                scope.reuse_variables()
                remainder_out = fn(remainder_inp)
                out = tf.concat([
                    tf.reshape(majority_out,
                        tf.concat([[leng - remainder], tf.shape(majority_out)[2:]], 0)),
                    remainder_out], 0)
            else:
                out = fn(inp)
        return tf.identity(out, name)
