#!/usr/bin/env python3

import sys
import os
import tensorflow as tf
import audio
import re
import argparse
import wave
import numpy as np

def get_flags():
    parser = argparse.ArgumentParser(description='Calculate audio features for a dataset')
    parser.add_argument('--channel', dest='channel', type=int, default=-1, choices=[-1, 0, 1],
            help="Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (int, default = -1)")
    parser.add_argument('--frame-length', dest='frame_length', type=int, default=25,
            help="Frame length in milliseconds (float, default=25)")
    parser.add_argument('--frame-shift', dest='frame_shift', type=int, default=10,
            help="Frame shift in milliseconds (float, default=10)")
    parser.add_argument('--output-format', dest='output_format', type=str, default="numpy",
            choices=["numpy", "tensorflow"],
            help="Output format (numpy or tensorflow) (str, default=numpy)")
    parser.add_argument('--scp', dest='scp', type=str, default=None,
            help="Output directory for SCP files. If set, multiple data files are generated for each utterance and an SCP is written to the directory as well.")
    parser.add_argument('--preemphasis-coefficient', dest='preemphasis', type=float, default=0.97,
            help="Coefficient for use in signal preemphasis (float, default = 0.97)")
    parser.add_argument('--sample-frequency', dest='sample_rate', type=int, default=16000,
            help="Waveform data sample frequency (must match the waveform file, if specified there) (float, default=16000")
    parser.add_argument('--online', dest='online', action='store_const', const=True, default=False,
            help="Process the features in an online fashion (i.e. remove DC offset with filter) (boolean, default=false)")
    parser.add_argument('--window-type', dest='window_type', type=str, default='hamming',
            choices=['hamming'],
            help='Type of window ("hamming") (string, default="hamming")')
    parser.add_argument('--nfft', dest='nfft', type=int, default=None,
            help='N_fft: number of points to use for FFT (defaults to next power of 2 over frame length)')
    parser.add_argument('--features', dest='features', type=str, nargs='+', default=['fbank23', 'logenergy'],
            help='Array of features to use. Default: --features fbank23 logenergy. (Mel filterbank with 23 features, log frame energy). Other options: mfcc23-13 (use mel filterbank with 23 features and get first 13 MFCCs), spec')
    parser.add_argument('-i', '--infile', dest='infile', type=str, default='/dev/stdin',
            help='Input SCP file. (string, default="/dev/stdin")')
    parser.add_argument('-o', '--outfile', dest='outfile', type=str, default='/dev/stdout',
            help='Output file if --scp is NOT set. (string, default="/dev/stdout")')
    return parser.parse_args()

class FbankFeat:
    def __init__(self, feature_count):
        self.feature_count = feature_count

    def get_tensor(self, aud):
        return aud.s_pe.log_mel_fbank_features

class MfccFeat:
    def __init__(self, feature_count, mfcc_count):
        self.feature_count = feature_count
        self.mfcc_count = mfcc_count

    def get_tensor(self, aud):
        return aud.s_pe.mfccs
class SpecFeat:
    def get_tensor(self, aud):
        return aud.s_pe.log_energy_spectrogram

class LogEnergyFeat:
    def get_tensor(self, aud):
        return tf.expand_dims(aud.s_pe.frame_energy_db, axis=2)

def process_features(config):
    mfcc_size = None
    filterbank_size = None
    features = []
    for feat in config.features:
        match1 = re.match(r"^fbank(\d+)$", feat)
        if match1 is not None:
            feature_count = int(match1.group(1))
            features.append(FbankFeat(feature_count))
            if filterbank_size is not None and filterbank_size != feature_count:
                raise RuntimeError("Inconsistant fbank sizes requested: %d and %d" % (filterbank_size, feature_count))
            filterbank_size = feature_count
            continue
        match2 = re.match(r"^mfcc(\d+)-(\d+)$", feat)
        if match2 is not None:
            feature_count = int(match2.group(1))
            mfcc_count = int(match2.group(2))
            features.append(MfccFeat(feature_count, mfcc_count))
            if filterbank_size is not None and filterbank_size != feature_count:
                raise RuntimeError("Inconsistant fbank sizes requested: %d and %d" % (filterbank_size, feature_count))
            if mfcc_size is not None and mfcc_size != mfcc_count:
                raise RuntimeError("Inconsistant mfcc sizes requested: %d and %d" % (mfcc_size, mfcc_count))
            filterbank_size = feature_count
            mfcc_size = mfcc_count
            continue
        if feat == "logenergy":
            features.append(LogEnergyFeat())
            continue
        if feat == "spec":
            features.append(SpecFeat())
            continue
        raise RuntimeError("Unrecognized feature: %s" % feat)
    if filterbank_size is None:
        filterbank_size = 23
    if mfcc_size is None:
        mfcc_size = 13
    return features, filterbank_size, mfcc_size

def get_feature_tensor(config, feature_schema, aud):
    with tf.name_scope("feature-tensor"):
        features = None
        for feat_schema in feature_schema:
            tensor = feat_schema.get_tensor(aud)
            if features is None:
                features = tensor
            else:
                features = tf.concat([features, tensor], axis=2)
        return tf.identity(features, "feature-tensor")

def process(config):
    feature_schema, fbank_size, mfcc_size = process_features(config)
    
    g = tf.Graph()
    with g.as_default():
        raw_waveforms = tf.placeholder(tf.float64, [1, None, 1], name="raw_waveforms")
        raw_waveform_lengths = tf.placeholder(tf.int32, [1], name="raw_waveform_lengths")
        aud = audio.AudioPreprocessing(raw_waveforms, raw_waveform_lengths,
                config.sample_rate, float(config.frame_length), float(config.frame_shift),
                channels=1, online=config.online,
                filterbank_size=fbank_size, mfcc_size=mfcc_size, preemphasis=config.preemphasis,
                N_fft=config.nfft)
        feature_tensor = get_feature_tensor(config, feature_schema, aud)
        if config.scp is None:
            if config.output_format == "numpy":
                out = {}
            elif config.output_format == "tensorflow":
                raise RuntimeError("Not yet implemented.")
        else:
            outfile = os.path.join(os.path.abspath(config.scp), "set.scp")
            scpout = open(outfile, "w")
        mean = 0
        mean_stddev = 0
        total = 0
        with tf.Session() as sess:
            with open(config.infile, "r") as f:
                for line in f:
                    uttid, wavfile = tuple(line.strip().split(" ", 1))
                    with wave.open(wavfile, "rb") as w:
                        assert w.getframerate() == config.sample_rate, "%d != %d" % (w.getframerate(), config.sample_rate)
                        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                        x = x.astype(np.float64) / float(np.iinfo(np.int16).max)
                    feats = sess.run(feature_tensor, feed_dict={
                        raw_waveforms: np.expand_dims(np.expand_dims(x, 0), -1),
                        raw_waveform_lengths: [x.shape[0]]
                    })
                    mean += np.mean(feats)
                    mean_stddev += np.std(feats)
                    total += 1
                    if config.scp is not None:
                        if config.output_format == "numpy":
                            outfile = os.path.join(os.path.abspath(config.scp), "%s.npy" % uttid)
                            np.save(outfile, feats)
                            scpout.write("%s %s\n" % (uttid, outfile))
                            scpout.flush()
                        elif config.output_format == "tensorflow":
                            raise RuntimeError("Not yet implemented.")
                    else:
                        if config.output_format == "numpy":
                            out[uttid] = feats
                        elif config.output_format == "tensorflow":
                            raise RuntimeError("Not yet implemented.")
        mean /= float(total)
        mean_stddev /= float(total)
        if config.scp is None:
            if config.output_format == "numpy":
                np.save(config.outfile, {"mean": mean, "mean_stddev": mean_stddev, "set": out})
            elif config.output_format == "tensorflow":
                raise RuntimeError("Not yet implemented.")
        else:
            scpout.close()
            outfile = os.path.join(os.path.abspath(config.scp), "means.npy")
            np.save(outfile, {"mean": mean, "mean_stddev": mean_stddev})

if __name__ == "__main__":
    process(get_flags())
