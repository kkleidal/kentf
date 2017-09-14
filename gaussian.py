import tensorflow as tf
from kentf.scoping import adapt_name

class DiagonalCovarianceGaussian:
    def __init__(self, means, log_variances, name=None):
        name = adapt_name(name, "gaussian")
        with tf.name_scope(name):
            self.name = name
            self.N = tf.identity(tf.shape(means)[0], "N")
            self.dims = tf.shape(means)[1:]
            self.means = tf.identity(means, "mean")
            self.means_with_L = tf.expand_dims(means, 1)
            self.log_variances = tf.identity(log_variances, "log-variance")
            self.log_variances_with_L = tf.expand_dims(log_variances, 1)
            self.stddevs = tf.identity(tf.exp(0.5 * self.log_variances), "stddev")
            self.variances = tf.identity(tf.exp(self.log_variances), "variance")
            self.variances_with_L = tf.expand_dims(self.variances, 1)

    def unit_gaussian(self, name=None):
        name = adapt_name(name, "unit-gaussian")
        with tf.name_scope(name):
            return DiagonalCovarianceGaussian(
                    tf.zeros(tf.shape(self.means)),
                    tf.ones(tf.shape(self.log_variances)), name=name)

    def sample(self, L=1, name=None):
        name = adapt_name(name, "sample_%s" % self.name)
        with tf.name_scope(name):
            shape = tf.concat([[self.N, L], self.dims], axis=0)
            noise = tf.random_normal(shape, 0, 1, dtype=tf.float32, name="noise")
            samples = self.means_with_L + (self.log_variances_with_L * noise)
            return tf.identity(samples, name)

    def log_likelihoods(self, samples, name=None):
        name = adapt_name(name, "log-likelihood_%s" % self.name)
        with tf.name_scope(name):
            out = np.log(2 * np.pi) + self.log_variances_with_L
            out += tf.square(samples - self.means_with_L) / self.variances_with_L
            out *= -0.5
            return tf.identity(out, name)

    def kl_divergence_from_unit(self, name=None):
        name = adapt_name(name, "kl-divergence-from-unit_%s" % self.name)
        return DiagonalCovarianceGaussian.kl_divergence(self,
            self.unit_gaussian(), name=name)

    @classmethod
    def kl_divergence(cls, p, q, name=None):
        name = adapt_name(name, "kl-divergence")
        with tf.name_scope(name):
            inner = p.variances + tf.square(p.means - q.means)
            inner /= q.variances
            inner = 1 + p.log_variances - q.log_variances - inner
            inner *= -0.5 
            kl = tf.reduce_sum(inner, list(range(1, len(inner.shape))))
            kl = tf.identity(kl, name)
        return kl

    @classmethod
    def symmetric_kl_divergence(cls, p, q, name=None):
        name = adapt_name(name, "symmetric-kl-divergence")
        with tf.name_scope(name):
            return tf.identity(cls.kl_divergence(p, q) + cls.kl_divergence(q, p), name)
