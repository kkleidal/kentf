import tensorflow as tf

def call_tracker(name, cache={}):
    fullname = "%s/%s" % (tf.get_default_graph().get_name_scope(), name)
    if fullname in cache:
        cache[fullname] += 1
        return "%s_%d" % (name, cache[fullname])
    else:
        cache[fullname] = 0
        return name

def adapt_name(name, default, cache={}, numbering=True):
    if name is not None:
        return name
    elif numbering:
        return default
    else:
        return call_tracker(default, cache=cache)
