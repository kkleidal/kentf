import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import scipy.io
import scoping

BATCH_AXIS = 0     # B
LENGTH_AXIS = 1    # L
DEPTH_AXIS = 2     # D
CHANNELS_AXIS = -1 # C


def sliding_window(inp, frame_length, frame_shift, max_number_frames=None, padding="VALID", name=None):
    '''
    Runs a sliding window across a signal (of audio data, for example).
    Params:
      - inp:  A signal tensor in BLC format where L is the number of SAMPLES
      - frame_length:  The length of each frame in number of samples (a python integer)
      - frame_shift:  How many samples to shift the window (a python integer)
      - padding:  How to pad the ends, can be "SAME" or "VALID"
    Returns:
      - A BLDC signal tensor where L is the number of FRAMES and D is frame_length.
    '''
    assert(len(inp.shape) == 3)  # BLC
    name = scoping.adapt_name(name, "sliding_window")
    with tf.name_scope(name):
        expanded = tf.expand_dims(inp, 3)
        lengths = [1, 1, 1, 1]
        shifts = [1, 1, 1, 1]
        lengths[LENGTH_AXIS] = frame_length
        shifts[LENGTH_AXIS] = frame_shift
        # Window the signal
        frames = tf.extract_image_patches(expanded, lengths, shifts, [1, 1, 1, 1], padding)
        if max_number_frames != None:
            # Clip the signal to only be the first max_number_frames frames
            slice_lengths = [-1 if i != LENGTH_AXIS else tf.cast(max_number_frames, tf.int32) for i in range(4)]
            frames = tf.slice(frames, [0, 0, 0, 0], tf.stack(slice_lengths))
        frames = tf.transpose(frames, [0, 1, 3, 2]) # BLCD --> BLDC
        frames = tf.identity(frames, name=name)
    return frames

class UnsupportedWindowTypeException(Exception):
    pass

def magnitude(complex_spec, name=None):
    '''
    Get the raw magnitude spectrogram for a complex spectrogram
    '''
    name = scoping.adapt_name(name, "magnitude")
    with tf.name_scope(name):
        return tf.abs(complex_spec, name=name)

def energy(complex_spec, name=None):
    '''
    Get the raw energy spectrogram for a complex spectrogram
    '''
    name = scoping.adapt_name(name, "energy")
    with tf.name_scope(name):
        return tf.cast(complex_spec * tf.conj(complex_spec), tf.float64, name=name)

def decibels(signal, name=None):
    '''
    Get the number of decibels (10 * log10(signal))) for a tensor of raw magnitudes
    '''
    name = scoping.adapt_name(name, "decibels")
    with tf.name_scope(name):
        return 10 * tf.maximum(tf.log(signal) / np.log(10), -50, name=name)

def get_Nfft(frame_length):
    # Get the next power of 2 above frame_length
    return int(np.power(2, np.ceil(np.log(np.float64(frame_length)) / np.log(2))))

def timeseries_to_spec(frames, frame_length, window_type='hamming', N_fft=None, zero_pad=True, name=None):
    '''
    Converts a timeseries to a spectrogram (preprocessing by removing the DC offset, zero padding,
        and applying a window function)
    Params:
        - frames: A BLDC tensor where L is the number of frames and D is the frame length
        - frame_length: python integer frame_length
        - window_type: the type of window (the same types supported as get_window)
        - N_fft: the number of FFT points to use (defaults to the next higher order of 2 after or equal to frame_length)
        - zero_pad: whether to zero_pad the frames to the next highest order of 2 for more efficient FFT
    Returns:
      N_fft, magnitude spec, energy spec, log magnitude spec (decibels), log energy spec (decibels)
      The spec is a BLDC tensor where L is the frame_length and D is the FFT bin count.
        FFT bin count is (N_fft or the next highest
        order of 2 above the frame_length) // 2 + 1 (to get the Nyquist frequency)
    '''
    name = scoping.adapt_name(name, "spec")
    with tf.name_scope(name):
        # The window to convolve the sample with
        window = tf.constant(get_window(window_type, frame_length).astype(np.float64), name="window")
        frames = tf.multiply(frames, tf.reshape(window, [1 if i != DEPTH_AXIS else -1 for i in range(4)]), name="windowing")
        # Padding/clipping to N_fft
        if zero_pad:
            if N_fft is None:
                N_fft = get_Nfft(frame_length)
            if N_fft > frame_length:
                # Pad the frames to N_fft
                padding = [[0, 0] if i != DEPTH_AXIS else [0, N_fft - frame_length] for i in range(4)]
                frames = tf.pad(frames, padding, "CONSTANT")
            elif N_fft < frame_length:
                # Downsample the frames to N_fft
                assert(DEPTH_AXIS == 2)
                frames = tf.image.resize_images(frames, [tf.shape(frames)[1], N_fft])
        ## New FFT
        #frames = tf.cast(tf.transpose(frames, [0, 1, 3, 2]), tf.float32) # BLDC -> BLCD
        #mag_spec = tf.spectral.rfft(frames, fft_length=[N_fft] if N_fft is not None else None)
        #mag_spec = tf.cast(tf.transpose(mag_spec, [0, 1, 3, 2]), tf.float64) # BLCD -> BLDC
        # FFT
        complex_frames = tf.complex(tf.cast(frames, tf.float32), tf.zeros(tf.shape(frames)))
        complex_frames = tf.transpose(complex_frames, [0, 1, 3, 2]) # BLDC -> BLCD
        spec = tf.fft(complex_frames)
        # Clip second half of spec:
        complex_spec = tf.slice(spec, tf.stack([0, 0, 0, 0]), tf.stack([-1, -1, -1, N_fft // 2 + 1]), name=name)
        complex_spec = tf.transpose(complex_spec, [0, 1, 3, 2]) # BLCD -> BLDC
        complex_spec = tf.cast(complex_spec, tf.complex128)
        mag_spec = magnitude(complex_spec, name="magnitude_spec")

        energy_spec = tf.square(mag_spec, name="energy_spec")
        log_mag_spec = decibels(mag_spec, name="log_magnetude_spec")
        log_energy_spec = 2 * log_mag_spec
        return N_fft, mag_spec, energy_spec, log_mag_spec, log_energy_spec

def apply_filterbank(spec, filter_bank, name=None):
    '''
    Params:
      - spec: BLDC where D is half the number of FFT bins (the lower half of the spec), the magnitude spectrum
      - filter_bank: [Half FFT bins, number filters] tensor, the filter bank 
    Returns:
      - BLDC tensor where D is the number of filters, the filter bank features
    '''
    name = scoping.adapt_name(name, "apply_filterbank")
    with tf.name_scope(name):
        shape = tf.shape(spec)
        assert(DEPTH_AXIS == 2)
        spec = tf.transpose(spec, [0, 1, 3, 2])                              # BLCD
        two_d = tf.reshape(spec, [-1, shape[DEPTH_AXIS]])             # *D
        feats = tf.matmul(two_d, filter_bank)                         # *D'
        feats = tf.reshape(feats, [shape[0], shape[1], shape[3], -1]) # BLCD'
        feats = tf.transpose(feats, [0, 1, 3, 2])                     # BLD'C
        return tf.identity(feats, name)

## UTILITY:
        
def get_window(window_type, N):
    '''
    Gets a window function to be applied to a timeseries via component-wise multiplication.
    Returns a numpy array which can be saved in the graph as a tf.constant.
    Params:
        - window_type: the type of window ('none', 'hamming', and 'hanning' supported)
        - N: the number of frames in the output
        - N_t: the number of frames in the input (only the first N of which will be used)
    Returns:
        - The window function of size N as a 1D np array
    Derived from a script by Michael Price
    '''
    result = np.zeros((N,))
    omega = np.linspace(0.0, 2.0 * np.pi, N)
    if window_type == 'none' or window_type is None:
        result = np.ones((N,))
    elif window_type == 'hamming':
        result = 0.54 - 0.46 * np.cos(omega)

    elif window_type == 'hanning':
        result = 0.5 - 0.5 * np.cos(omega)
    else:
        raise UnsupportedWindowTypeException()
    return result
        
def freq_to_mel(freq):
    '''
    Convert a frequency in Hz to a mel index
    '''
    return 1125.0 * np.log(1.0 + freq / 700.0)

def mel_to_freq(mel):
    '''
    Convert the mel index to a frequency in Hz
    '''
    return 700.0 * (np.exp(mel / 1125.0) - 1.0)

def mel_filterbank(N_fft, sample_rate, num_bands=23, low_freq=54, high_freq=None):
    '''
    Get the mel filterbank with the given dimensions as a numpy array.
    This can then be wrapped as a tf constant for use in tensorflow.
    '''
    if high_freq is None:
        high_freq = sample_rate / 2 - 1
    # Get the high frequency/low frequency for the bank
    high_freq = float(min(high_freq, sample_rate / 2))
    low_freq = float(max(0, low_freq))
    # Convert the extremes to mel
    high_mel = freq_to_mel(high_freq)
    low_mel = freq_to_mel(low_freq)
    # Get the bins for the filterbank by linearly spacing in mel space
    mels = np.linspace(low_mel, high_mel, num_bands + 2)
    # Convert the bins back to frequencies
    freqs = mel_to_freq(mels)
    # Convert the frequencies to FFT bins
    bins = np.ceil((freqs / float(sample_rate / 2) * (N_fft // 2))).astype(np.int32)
    # Make the triangular filters
    fbank = np.zeros((N_fft // 2 + 1, num_bands))
    for band in range(num_bands):
        fbank[bins[band]:bins[band+1]+1,band] = np.linspace(0.0, 1.0, bins[band + 1] - bins[band] + 2)[1:]
        fbank[bins[band+1]:bins[band+2]+1,band] = np.linspace(1.0, 0.0, bins[band + 2] - bins[band + 1] + 2)[:-1]
    return fbank.astype(np.float64)

def constantify(x, name="constant"):
    '''
    Create a tf constant that is initialized with x and the given name,
    returns x and the new constant
    '''
    return x, tf.constant(x, name=name)

def variableify(x, name="variable"):
    '''
    Create a tf variable that is initialized with x and the given name,
    returns x and the new variable
    '''
    return x, tf.Variable(x, name=name, trainable=False)
 
def dct_matrix(filterbank_size=23, mfcc_size=13):
    '''
    Get a DCT matrix for converting MFSCs -> MFCCs
    '''
    dct = np.zeros((mfcc_size, filterbank_size))
    for i in range(mfcc_size):
        for j in range(filterbank_size):
            dct[i, j] = np.cos(np.pi * np.float64(i) / np.float64(filterbank_size) * (np.float64(j) + 0.5))
    return dct.T

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
            
def remove_dc(signal, signal_lengths, signal_mask, last_sof=None, last_sin=None, online=True, name=None):
    name = scoping.adapt_name(name, "remove_dc")
    with tf.variable_scope(name):
        if online:
            batch_size = tf.shape(signal)[0]
            length = tf.shape(signal)[1]
            channels = tf.shape(signal)[-1]
            zeros = tf.zeros([batch_size, 1, channels], dtype=tf.float64)
            if last_sof is None:
                last_sof = zeros 
            if last_sin is None:
                last_sin = zeros
            sof = last_sof
            i = tf.constant(0)
            def body(i, sof, last_sin):
                last_sof = tf.slice(sof, [0, i, 0], [-1, 1, -1])
                cur_sin = tf.slice(signal, [0, i, 0], [-1, 1, -1])
                cur_mask = tf.slice(signal_mask, [0, i, 0], [-1, 1, -1])
                cur_sof = tf.where(
                    cur_mask,
                    cur_sin - last_sin + 0.999 * last_sof,
                    zeros)
                new_sof = tf.concat([sof, cur_sof], axis=1)
                newi = tf.add(i, 1)
                return newi, new_sof, cur_sin
            _, sof, _ = tf.while_loop(
                lambda i, sof, last_sin: tf.less(i, length),
                body,
                loop_vars=[i, sof, last_sin],
                shape_invariants=[
                    i.get_shape(),
                    tf.TensorShape([None, None, None]),
                    last_sin.get_shape()],
                back_prop=False)
            sof = tf.slice(sof, [0, 1, 0], [-1, -1, -1])
            sof = tf.reshape(sof, tf.shape(signal))
            demeaned = tf.identity(sof, "demeaned")
            mean = tf.identity(signal - demeaned, "mean")
            last_sin = tf.slice(last_sin, [0, tf.shape(signal)[1] - 1, 0], [-1, 1, -1])
            last_sof = tf.slice(demeaned, [0, tf.shape(demeaned)[1] - 1, 0], [-1, 1, -1])
            return mean, demeaned, last_sof, last_sin
        else:
            # If index < signal_lengths, count in the mean and remove mean
            zeros = tf.zeros(tf.shape(signal), dtype=tf.float64)
            mean = tf.reduce_sum(tf.where(signal_mask, signal, zeros), axis=LENGTH_AXIS, keep_dims=True) / tf.cast(tf.expand_dims(signal_lengths, 1), tf.float64)
            mean = tf.identity(mean, "mean")
            demeaned = tf.where(signal_mask, signal - mean, zeros)
            return mean, demeaned, None, None



class AudioPreprocessing:
    '''
    For extracting features from an audio signal
    '''
    ms_per_sec = 1000
    
    def __init__(self, raw_waveforms, raw_waveform_lengths,
            sample_rate, frame_length_ms, frame_shift_ms,
            last_sof=None, last_sin=None, online=True,
            channels=1, filterbank_size=23, mfcc_size=13, preemphasis=0.97, max_length=None, N_fft=512,
            name=None):
        '''
        Preprocess audio, setting ops to remove DC offset, window the function, perform FFT, calculate 
            mel filterbanks and mfccs, etc.
        Params:
            - raw_waveforms: a BLC signal tensor in float format with values from -1 to 1. L is the number of samples.
            - raw_waveform_lengths: a B integer tensor with the length in samples of each example in the batch
            - sample_rate: a python integer. The sample rate for the raw_waveforms
            - frame_length_ms: a python integer. The frame length in milliseconds for signal processing
            - frame_shift_ms: a python integer. The frame shift in milliseconds for signal processing
            - channels: a python integer: the number of channels for the signal
            - filterbank_size: a python integer. The number of mel filters to use
            - mfcc_size: a python integer. The number of MFCCs to use
            - preemphasis: a python float. The preemphasis factor (between 0 and 1) to apply: c in Y = (1 - Rc)X
            - max_length: the max_length in samples for the raw_waveforms (the tails will be clipped if they exceed it); None for no max length
            - N_fft: the number of FFT points to use
        '''
        name = scoping.adapt_name(name, "audio_preprocessing")
        with tf.name_scope(name):
            with tf.name_scope("params"):
                def convert_ms_to_samples(ms, name="ms_to_samples"):
                    x = np.int32(float(ms) / float(ms_per_sec) * float(sample_rate))
                    return constantify(x, name)
                self.frame_length_py, self.frame_length = convert_ms_to_samples(frame_length_ms, name="frame_length")
                self.frame_shift_py, self.frame_shift = convert_ms_to_samples(frame_shift_ms, name="frame_shift")
                if N_fft is None:
                    N_fft = get_Nfft(self.frame_length_py)
                self.N_fft_py, self.N_fft = constantify(N_fft, "N_fft")
                self.raw_waveforms = raw_waveforms
                self.raw_waveform_lengths = raw_waveform_lengths
                self.sample_rate_py, self.sample_rate = constantify(sample_rate, name="sample_rate")
                self.frame_length_ms_py, self.frame_length_ms = constantify(frame_length_ms, "frame_length_ms")
                self.frame_shift_ms = constantify(frame_shift_ms, "frame_shift_ms")
                self.channels_py, self.channels = constantify(channels, "channels")
                self.preemphasis_py, self.preemphasis = constantify(np.float64(preemphasis), "preemphasis")
                self.filterbank_size_py, self.filterbank_size = constantify(filterbank_size, "filterbank_size")
                self.mfcc_size_py, self.mfcc_size = constantify(mfcc_size, "mfcc_size")
                preemphasis, preemphasis_py = variableify(preemphasis, name="preemphasis")
            with tf.name_scope("mask"):
                idxes = tf.tile(
                    tf.expand_dims(tf.range(0, tf.shape(self.raw_waveforms)[LENGTH_AXIS], 1, dtype=tf.int32), 0),
                    [tf.shape(self.raw_waveforms)[BATCH_AXIS], 1])
                self.mask = idxes < tf.expand_dims(self.raw_waveform_lengths, 1)

                self.mask = tf.tile(tf.expand_dims(self.mask, -1),
                        [1, 1, tf.shape(self.raw_waveforms)[-1]])
            self.dc_offset, self.s_of, self.last_sof, self.last_sin = remove_dc(self.raw_waveforms,
                    self.raw_waveform_lengths, self.mask,
                    last_sof=last_sof, last_sin=last_sin, online=online, name="remove_dc")
            with tf.name_scope("preemphasis"):
                '''
                kernel = [
                    [[-1.0 * self.preemphasis if i == j else 0.0 for j in range(channels)]
                        for i in range(channels)],
                    [[1.0 if i == j else 0.0 for j in range(channels)]
                        for i in range(channels)]]
                self.s_pe = tf.nn.conv1d(
                        tf.cast(self.s_of, tf.float32),
                        tf.cast(kernel, tf.float32),
                        1, "SAME")
                self.s_pe = tf.cast(self.s_pe, tf.float64)
                '''
                zeros =  tf.zeros([tf.shape(self.s_of)[0], 1, self.channels_py], dtype=tf.float64)
                s_of   = tf.concat([self.s_of, zeros], axis=LENGTH_AXIS)
                s_of_R = tf.concat([zeros, self.s_of], axis=LENGTH_AXIS)
                s_pe = s_of - self.preemphasis * s_of_R
                s_pe = tf.slice(s_pe, [0, 0, 0], tf.stack([-1, tf.shape(s_pe)[1] - 1, -1]))
                self.s_pe = tf.identity(s_pe, name="s_pe")
            with tf.name_scope("padding"):
                self.padding = self.frame_length - tf.mod(self.raw_waveform_lengths - self.frame_length, self.frame_shift)
                self.padding = tf.where(tf.equal(self.padding, self.frame_length), 0 * self.padding, self.padding, name="padding")
                self.padded_waveform_lengths = tf.add(self.raw_waveform_lengths, self.padding, name="padded_waveform_lengths")
                self.frame_counts = tf.divide(self.padded_waveform_lengths - self.frame_length, self.frame_shift, name="frame_counts")
                self.max_padded_length = tf.reduce_max(self.padded_waveform_lengths, name="max_padded_length")
                self.total_padding = tf.identity(self.max_padded_length - tf.reduce_max(self.raw_waveform_lengths), name="total_padding")
            self.filterbank = tf.constant(mel_filterbank(self.N_fft_py, self.sample_rate_py,
                    num_bands=self.filterbank_size_py), name="filterbank")
            self.dct_matrix = tf.constant(dct_matrix(self.filterbank_size_py, self.mfcc_size_py).astype(np.float64), name="dct_matrix")
            self.s_of = SignalPreprocessing(self, self.s_of, name="s_of")
            self.s_pe = SignalPreprocessing(self, self.s_pe, name="s_pe")
            self.aurora_features = tf.concat([
                tf.expand_dims(self.s_of.frame_energy, DEPTH_AXIS),
                self.s_pe.mfccs], axis=DEPTH_AXIS, name="aurora_features")

class SignalPreprocessing:
    '''
    Ops for processing a signal: padding it, windowing it, performing FFT, and then applying a filterbank/dct.
        Assumes DC offset and preemphasis has already been applied.
    Params:
        - audioPreprocessing: the parent AudioPreprocessing object
        - signal: the BLC float tensor with values between -1 and 1 representing the raw waveform to process
    '''
    def __init__(self, audioPreprocessing, signal, name=None):
        name = scoping.adapt_name(name, "signal")
        with tf.name_scope(name):
            self.audio = audioPreprocessing
            self.signal = signal
            # Pad:
            with tf.name_scope("padding"):
                self.signal_padded = tf.pad(self.signal,
                    tf.stack([
                        tf.stack([0, 0]) if i != LENGTH_AXIS else tf.stack([0, self.audio.total_padding])
                            for i in range(3)
                    ]),
                    name="signal_padded")
                self.windowed = sliding_window(self.signal_padded, self.audio.frame_length_py, self.audio.frame_shift_py,
                    max_number_frames=tf.reduce_max(self.audio.frame_counts), name="windowed_frames")
                self.frame_energy = tf.identity(
                    tf.maximum(
                        np.float64(-50.0),
                        tf.log(tf.reduce_sum(tf.square(self.windowed), axis=[DEPTH_AXIS]))), # / tf.log(10.0)),
                    name="frame_energy")
                self.frame_energy_db = tf.identity(10.0 * self.frame_energy, name="frame_energy_db")
            # Spec:
            _, self.magnitude_spectrogram, self.energy_spectrogram, \
                    self.log_magnitude_spectrogram, self.log_energy_spectrogram = \
                    timeseries_to_spec(self.windowed, self.audio.frame_length_py,
                                       window_type='hamming', N_fft=self.audio.N_fft_py,
                                       zero_pad=True, name="spectrogram")
            # Fbanks:
            self.mel_fbank_features = apply_filterbank(self.energy_spectrogram,
                    self.audio.filterbank, name="mel_fbank_features")
            self.mel_fbank_features_from_magnitude = apply_filterbank(self.magnitude_spectrogram,
                    self.audio.filterbank, name="mel_fbank_features_from_magnitude")
            self.log_mel_fbank_features = decibels(self.mel_fbank_features, name="log_mel_fbank_features")
            # MFCCs:
            with tf.name_scope("mfccs"):
                self.mfscs = tf.maximum(tf.log(self.mel_fbank_features), -50, name="mfscs")
                mfscs = tf.transpose(self.mfscs, [0, 1, 3, 2]) # BLDC -> BLCD
                mfscs_flat = tf.reshape(mfscs, [-1, self.audio.filterbank_size_py])
                mfccs_flat = tf.matmul(mfscs_flat, self.audio.dct_matrix)
                shape = tf.concat([tf.slice(tf.shape(mfscs), [0], [3]), [self.audio.mfcc_size]],
                    axis=0, name="mfccs_shape")
                mfccs = tf.reshape(mfccs_flat, shape)
                mfccs = tf.transpose(mfccs, [0, 1, 3, 2]) # BLCD -> BLDC
                mfccs = tf.identity(mfccs, "mfccs")
                self.mfccs = mfccs

def read_wav_audio(filename):
    '''
    Read a wavefile into memory, returns the sample rate (in Hz) and the input signal
    '''
    sample_rate, data = scipy.io.wavfile.read(filename)
    # Convert samples from S16 -> F32
    s_in = data.astype(np.float64) / float(np.iinfo(np.int16).max)
    return sample_rate, s_in

ms_per_sec = 1000
            
def example():
    import matplotlib.pyplot as plt
    import specplotting

    sample_rate, s_in = read_wav_audio("./scratch/lab1-resources/gas_station.wav")
    samples = len(s_in)
    print("The file is %d samples long" % samples);
    print('The sample rate is %d Hz' % sample_rate);

    ms_per_sec = 1000.0;
    milliseconds = samples / sample_rate * ms_per_sec;
    print('The file is %d milliseconds long' % milliseconds);

    ref_mfccs = scipy.io.loadmat("./scratch/lab1-resources/gas_station.ref.new.mat")["mfcc_ref"]
    ref_mfccs = np.expand_dims(ref_mfccs, 0)
    ref_mfccs = np.expand_dims(ref_mfccs, 3)
    print(ref_mfccs.shape)

    ref_magspec = scipy.io.loadmat("./scratch/magspec.mat")["magspec"]
    ref_magspec = np.transpose(ref_magspec, [1, 0])
    print("magspec", ref_magspec.shape)

    ref_spe = scipy.io.loadmat("./scratch/spe.mat")["s_pe"]
    print("spe", ref_spe.shape)

    ref_sof = scipy.io.loadmat("./scratch/s_of_ref.mat")["s_of"]
    print("sof", ref_sof.shape)

    ref_melfilt = scipy.io.loadmat("./scratch/fbanks.mat")["melfiltered"]
    ref_melfilt = np.transpose(ref_melfilt, [1, 0])

    ref_fbank = scipy.io.loadmat("./scratch/lab1-resources/mel_filters.mat")["mel_filters"]

    inp = np.reshape(s_in, [1, s_in.shape[0], 1])
    inp = np.concatenate([inp, -2 * inp], axis=2)
    inplens = np.array([s_in.shape[0]])
    print(inp.shape)
    print(inplens)

    g = tf.Graph()
    with g.as_default():
        raw_waveforms = tf.placeholder(tf.float64, [None, None, 2], name="raw_waveforms")
        raw_waveform_lengths = tf.placeholder(tf.int32, [None], name="raw_waveform_lengths")
        audio = AudioPreprocessing(raw_waveforms, raw_waveform_lengths, 16000, 25.0, 10.0, channels=2, online=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                g.get_tensor_by_name("raw_waveforms:0"): inp,
                g.get_tensor_by_name("raw_waveform_lengths:0"): inplens,
            }
            print(sess.run(tf.shape(audio.s_pe.windowed), feed_dict=feed_dict))
            print(sess.run(tf.shape(audio.s_of.windowed), feed_dict=feed_dict))
            print(sess.run(tf.shape(audio.s_pe.frame_energy), feed_dict=feed_dict))
            print(sess.run(tf.shape(audio.s_pe.magnitude_spectrogram), feed_dict=feed_dict))
            out = sess.run({
                "dc_off": audio.dc_offset,
                "frame_length": audio.frame_length,
                "frame_shift": audio.frame_shift,
                "padding": audio.padding,
                "padded_length": audio.padded_waveform_lengths,
                "total_padding": audio.total_padding,
                "frame_counts": audio.frame_counts,
                "windowed_frames_s_of": audio.s_of.windowed,
                "windowed_frames_s_pe": audio.s_pe.windowed,
                "frame_energy_s_of": audio.s_of.frame_energy,
                "frame_energy_s_pe": audio.s_pe.frame_energy,
                "spec_no_pe": audio.s_of.log_energy_spectrogram,
                "spec": audio.s_pe.log_energy_spectrogram,
                "spec_mag": audio.s_pe.energy_spectrogram,
                "fbank": audio.filterbank,
                "raw_filterbank": audio.s_pe.mel_fbank_features,
                "filterbank": audio.s_pe.log_mel_fbank_features,
                "aurora": audio.aurora_features,
                "s_pe": audio.s_pe.signal_padded,
                "s_of": audio.s_of.signal,
            }, feed_dict=feed_dict)
            print("The DC offset was: %(dc_off)s" % out)
            print("The frame length was: %(frame_length)d" % out)
            print("The frame shift was: %(frame_shift)d" % out)
            print("The padding was: %(padding)d" % out)
            print("The total padding was: %(total_padding)s" % out)
            print("There were %(frame_counts)d frames" % out)
            print("The padded length was %(padded_length)d samples" % out)
            print("The framed audio was %(windowed_frames_s_of)s" % out)
            print("The framed audio was %(windowed_frames_s_pe)s" % out)
            print("The framed audio energy for the 50th frame was %s" % out["frame_energy_s_of"][0, 49, :])
            print("The framed audio energy for the 50th frame was %s" % out["frame_energy_s_pe"][0, 49, :])
            sof_error = np.square(out["s_of"][0,:,0] - ref_sof[:,0])
            ma_sof_error = moving_average(sof_error, 400)
            print("Average s_of error: %.f", np.mean(sof_error))
            plt.title("$s_{of}$ Error$")
            plt.plot(np.arange(ma_sof_error.shape[0]), ma_sof_error) 
            plt.show()
            spe_error = np.square(out["s_pe"][0,:,0] - ref_spe[:,0])
            ma_spe_error = moving_average(spe_error, 400)
            print("Average s_pe error: %.f", np.mean(spe_error))
            plt.title("$s_{pe}$ Error$")
            plt.plot(np.linspace(0,ma_spe_error.shape[0],ref_spe.shape[0]), ref_spe[:,0]) 
            ax2 = plt.gca().twinx()
            ax2.plot(np.arange(ma_spe_error.shape[0]), ma_spe_error, c="r") 
            plt.show()
            plt.title("Energy")
            plt.plot(range(int(out["frame_counts"][0])), out["frame_energy_s_of"][0, :, 0])
            plt.xlabel("Frame Number")
            plt.ylabel("Energy (dB)")
            plt.show()
            print("There were %d frames" % out["frame_energy_s_of"].shape[2])
            print("Spec: %(spec)s" % out)
            print("Spec size: %s" % str(out["spec"].shape))
            specplotting.plot_spec(out["spec"][0,:,:,0], sample_rate=sample_rate, title="Energy Spectrogram (dB)")
            plt.show()
            specplotting.plot_spec(out["spec_no_pe"][0,:,:,0], sample_rate=sample_rate, title="Energy Spectrogram (dB) No Pre-Emphasis")
            plt.show()
            specplotting.plot_spec(out["spec_mag"][0,:,:,0], sample_rate=sample_rate, title="Raw Energy Spectrogram")
            plt.show()
            specplotting.plot_spec(ref_magspec, sample_rate=sample_rate, title="Ref Raw Energy Spectrogram")
            plt.show()
            specplotting.plot_spec(np.square(out["spec_mag"][0,:,:,0] - ref_magspec), sample_rate=sample_rate, title="Raw Energy Spec Error")
            plt.show()

            specplotting.plot_spec(out["fbank"], title="Fbank Matrix")
            plt.xlabel("FFT Bin")
            plt.ylabel("Filter")
            plt.show()
            specplotting.plot_spec(ref_fbank, title="Ref Fbank Matrix")
            plt.xlabel("FFT Bin")
            plt.ylabel("Filter")
            plt.show()
            specplotting.plot_spec(np.square(out["fbank"] - ref_fbank), title="Raw Ref Fbank Error")
            plt.xlabel("FFT Bin")
            plt.ylabel("Filter")
            plt.show()

            specplotting.plot_spec(out["raw_filterbank"][0,:,:,0], title="Raw Mel Filterbank")
            plt.show()
            specplotting.plot_spec(ref_melfilt, title="Ref Raw Mel Filterbank")
            plt.show()
            specplotting.plot_spec(np.square(out["raw_filterbank"][0,:,:,0] - ref_melfilt), title="Raw Mel Filterbank Error")
            plt.show()

            specplotting.plot_spec(out["filterbank"][0,:,:,0], title="Mel Filterbank Energies (dB)")
            plt.ylabel("Feature")
            plt.show()
            specplotting.plot_spec(out["aurora"][0,:,:,0], title="Aurora Features")
            plt.ylabel("Feature")
            plt.show()
            specplotting.plot_spec(ref_mfccs[0,:,:,0], title="Reference Aurora Features")
            plt.ylabel("Feature")
            plt.show()
            err = np.square(ref_mfccs[0,:-1,:,0] - out["aurora"][0,:-1,:,0])
            specplotting.plot_spec(err, title="Aurora Features Squared Error")
            plt.ylabel("Feature")
            plt.show()
            mfcc = 1
            X = np.arange(out["aurora"].shape[LENGTH_AXIS])
            plt.plot(X, out["aurora"][0,:,mfcc:mfcc+1,0])
            plt.show()
            plt.plot(ref_mfccs[0,:,mfcc:mfcc+1,0])
            plt.show()
            plt.plot(X[:-1], err[:,mfcc:mfcc+1])
            plt.show()

def example_librosa():
    import matplotlib.pyplot as plt
    import specplotting
    import librosa

    audio_path = "./scratch/lab1-resources/gas_station.wav"
    sample_rate, s_in = read_wav_audio(audio_path)
    lr_y, lr_sr = librosa.load(audio_path, 16000)
    print("Librosa sr: ", lr_sr)
    samples = len(s_in)
    print("The file is %d samples long" % samples);
    print('The sample rate is %d Hz' % sample_rate);

    ms_per_sec = 1000.0;
    milliseconds = samples / sample_rate * ms_per_sec;
    print('The file is %d milliseconds long' % milliseconds);

    inp = np.reshape(s_in, [1, s_in.shape[0], 1])
    inplens = np.array([s_in.shape[0]])

    g = tf.Graph()
    with g.as_default():
        raw_waveforms = tf.placeholder(tf.float64, [None, None, 1], name="raw_waveforms")
        raw_waveform_lengths = tf.placeholder(tf.int32, [None], name="raw_waveform_lengths")
        N_fft=512
        audio = AudioPreprocessing(raw_waveforms, raw_waveform_lengths, 16000, 25.0, 10.0, N_fft=N_fft, channels=1)

        print(audio.frame_length_py)
        print(audio.N_fft_py)
        print(audio.frame_shift_py)

        S = librosa.core.stft(lr_y, n_fft=N_fft, hop_length=audio.frame_shift_py, win_length=audio.frame_length_py,
                window="hamming", center=True, pad_mode="constant")
        print(S.shape)
        # S = librosa.feature.melspectrogram(S=S, sr=lr_sr, n_mels=23)
        ref_fbank = librosa.logamplitude(S, amin=10**(-50)).T

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                g.get_tensor_by_name("raw_waveforms:0"): inp,
                g.get_tensor_by_name("raw_waveform_lengths:0"): inplens,
            }
            out = sess.run({
                "filterbank": audio.s_pe.log_magnitude_spectrogram, # audio.s_pe.log_mel_fbank_features,
            }, feed_dict=feed_dict)
            print(out["filterbank"][0,:,:,0].shape)
            specplotting.plot_spec(out["filterbank"][0,:,:,0], sample_rate=sample_rate, title="Mel Filterbank Energies (dB)")
            plt.ylabel("Feature")
            plt.show()
            specplotting.plot_spec(ref_fbank, sample_rate=sample_rate, title="Librosa Ref Mel Filterbank Energies (dB)")
            plt.ylabel("Feature")
            plt.show()
