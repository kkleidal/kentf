import tensorflow as tf
import tflearn as tfl
from kentf.scoping import adapt_name

def merge_grads(grad_list, name=None):
    name = adapt_name(name, "merge-grads")
    with tf.name_scope(name):
        for i in range(len(grad_list)):
            grad_list[i].sort(key=lambda x: x[1].name)
        out = []
        for i, (tensor, vari) in enumerate(grad_list[0]):
            names = [grad_list[j][i][1].name for j in range(len(grad_list))]
            assert all((name == vari.name for name in names)), names
            grad_avg = tf.reduce_mean(
                tf.stack([grad_list[j][i][0] for j in range(len(grad_list))], axis=0),
                axis=0)
            out.append((grad_avg, vari))
        return out

def split_and_recombined(inps, fn, num_splits, name=None):
    name = adapt_name(name, "split-and-recombine")
    with tf.name_scope(name):
        adapted_inps = []
        # Split inputs:
        with tf.name_scope("preprocessing"):
            for inp in inps:
                if isinstance(inp, list) or isinstance(inp, tuple):
                    if len(inp) % num_splits != 0:
                        raise RuntimeError("List not divisible by number of splits: %s" % repr(inp))
                    stride = len(inp) // num_splits
                    squeeze = lambda x: x[0] if len(x) == 1 else x
                    adapted_inps.append([squeeze(inp[i:(i+stride)]) for i in range(0, len(inp), stride)])
                elif (isinstance(inp, tf.Variable) or isinstance(inp, tf.Tensor))\
                        and len(inp.shape) > 0:
                    if inp.shape[0].value is None:
                        raise RuntimeError("Batch index must be defined for tensor")
                    leng = int(inp.shape[0])
                    if leng % num_splits != 0:
                        raise RuntimeError("Tensor not divisible by number of splits (%d): %s" % (num_splits, inp.shape))
                    stride = leng // num_splits
                    adapted_inps.append([
                        tf.slice(inp,
                            [i if j == 0 else 0 for j in range(len(inp.shape))],
                            [stride if j == 0 else -1 for j in range(len(inp.shape))])
                        for i in range(0, leng, stride)])
                else:
                    adapted_inps.append([inp] * num_splits)
        # Zip inputs to divide work:
        adapted_inps = list(zip(*adapted_inps))
        # Do work
        raw_outputs = []
        for split, args in enumerate(adapted_inps):
            with tf.name_scope("bin%d" % split):
                raw_outputs.append(fn(*args))
        # Post-process outputs
        outputs = []
        with tf.name_scope("postprocessing"):
            for i, group in enumerate(raw_outputs):
                for j, var in enumerate(group):
                    if i == 0:
                        outputs.append([var])
                    else:
                        outputs[j].append(var)
        return outputs

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
