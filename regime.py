import tensorflow as tf
from kentf.scoping import adapt_name

class RegimeSpan:
    def __init__(self, start_epoch, end_epoch, values):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.values = values

class Regime:
    spans = [RegimeSpan(0, 1, {})]
    defaults = {}

    def lookup_row(self, epoch, name=None):
        name = adapt_name(name, "lookup-regime-row")
        def get_regime_val(span, key):
            return lambda: span.value.get(key, self.defaults[key])
        def get_key(key):
            cases = [
                (tf.logical_and(epoch >= span.start_epoch, epoch <  span.end_epoch),
                    get_regime_val(span, key))
                for span in self.spans]
            chosen = tf.case(cases, lambda: self.defaults[key])
            return tf.identity(chosen, "regime-%s" % key)
        return {key: get_key(key) for key in self.defaults}
