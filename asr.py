import tensorflow as tf
import numpy as np

def sliding_window(inp, frame_length, frame_shift, padding="VALID", name=None):
    '''
    Runs a sliding window across a signal (of audio data, for example).
    Params:
      - inp:  A [batch_size, channels, signal_length] tensor
      - frame_length:  The length of each frame in number of samples
      - frame_shift:  How many samples to shift the window
      - padding:  How to pad the ends, can be "SAME" or "VALID"
    Returns:
      - A [batch_size, channels, signal_length // frame_shift (approx, depending on padding), frame_length] tensor of frames

    Example:

    ```
    p = tf.placeholder(tf.float32, [None, 2, None])
    patches = sliding_window(p, 5, 3, padding="VALID")

    with tf.Session() as sess:
        # Make 3, 2-channel signals
        a1 = np.vstack([np.arange(15), np.arange(15, 15 + 15)])
        a2 = np.vstack([np.arange(30, 30 + 15), np.arange(45, 45 + 15)])
        a3 = np.vstack([np.arange(60, 60 + 15), np.arange(75, 75 + 15)])
        # Stack the batch:
        batch = np.vstack([np.expand_dims(a1, 0), np.expand_dims(a2, 0), np.expand_dims(a3, 0)])
        x = sess.run(patches, feed_dict={
            p: batch 
        })
        print(x.shape)
        print(x)
    ```

    Prints:
    (3, 2, 4, 5)
    [[[[  0.   1.   2.   3.   4.]
       [  3.   4.   5.   6.   7.]
       [  6.   7.   8.   9.  10.]
       [  9.  10.  11.  12.  13.]]

      [[ 15.  16.  17.  18.  19.]
       [ 18.  19.  20.  21.  22.]
       [ 21.  22.  23.  24.  25.]
       [ 24.  25.  26.  27.  28.]]]


     [[[ 30.  31.  32.  33.  34.]
       [ 33.  34.  35.  36.  37.]
       [ 36.  37.  38.  39.  40.]
       [ 39.  40.  41.  42.  43.]]

      [[ 45.  46.  47.  48.  49.]
       [ 48.  49.  50.  51.  52.]
       [ 51.  52.  53.  54.  55.]
       [ 54.  55.  56.  57.  58.]]]


     [[[ 60.  61.  62.  63.  64.]
       [ 63.  64.  65.  66.  67.]
       [ 66.  67.  68.  69.  70.]
       [ 69.  70.  71.  72.  73.]]

      [[ 75.  76.  77.  78.  79.]
       [ 78.  79.  80.  81.  82.]
       [ 81.  82.  83.  84.  85.]
       [ 84.  85.  86.  87.  88.]]]]
    '''
    assert(len(inp.shape) == 3)  # [batch_size, channels, frame length]
    with tf.name_scope(name or "sliding_window"):
        expanded = tf.expand_dims(inp, 3)
        frames = tf.extract_image_patches(expanded, [1, 1, frame_length, 1], [1, 1, frame_shift, 1], [1, 1, 1, 1], padding)
    return frames

class UnsupportedWindowTypeException(Exception):
    pass

def get_window(window_type, N, N_t=None):
    '''
    Gets a window function to be applied to a timeseries via component-wise multiplication
    Params:
        - window_type: the type of window ('none', 'hamming', and 'hanning' supported)
        - N: the number of frames in the output
        - N_t: the number of frames in the input (only the first N of which will be used)
    Returns:
        - The window function of size N as a 1D np array
    Derived from a script originally by Michael Price
    '''
    if N_t is None:
        N_t = N
    result = tf.zeros((N_t,))
    omega = tf.linspace(0.0, 2.0 * np.pi, N)
    if window_type == 'none' or window_type is None:
        result = np.ones((N,))
    elif window_type == 'hamming':
        result = 0.54 - 0.46 * tf.cos(omega)
    elif window_type == 'hanning':
        result = 0.5 - 0.5 * tf.cos(omega)
    else:
        raise UnsupportedWindowTypeException()
    return tf.pad(result, [[0, N_t - N]])

def magnitude(complex_spec):
    return tf.abs(complex_spec)

def power(complex_spec):
    return tf.power(magnitude(complex_spec), 2)

def decibels(complex_spec):
    return 20 * tf.log(magnitude(complex_spec)) / np.log(10)

def timeseries_to_spec(frames, window_type='hamming', zero_pad=True, remove_dc=True, gpu=False):
    '''
    Converts a timeseries to a spectrogram (preprocessing by removing the DC offset, zero padding,
        and applying a window function)
    Params:
        - frames: A [batch_size, channels, frame count, frame_length] tensor
        - window_type: the type of window (the same types supported as get_window)
        - zero_pad: whether to zero_pad the frames to the next highest order of 2 for more efficient FFT
        - remove_dc: whether to remove the DC offset (mean) of each frame
    Returns: A [batch_size, frame_count, FFT bin count] tensor.  FFT bin count is the next highest order of 2
        above the frame_length (divided by 2) if zero_pad is True and is the frame_length (divided by 2) if
        zero_pad is False.  The division by 2 is to get to the Nyquist frequency.
    Example:

    import scipy.io.wavfile as wavfile

    def spec_generator(audio, sample_rate):
        frame_length = int(0.025 * sample_rate)
        frame_shift = int(0.015 * sample_rate)
        frames = sliding_window(audio, frame_length, frame_shift, padding="VALID")
        spec = timeseries_to_spec(frames)
        spec = decibels(spec)
        return spec

    with tf.Session() as sess:
        sr, audio = wavfile.read("/data/sls/u/urop/kkleidal/three.wav")
        p = tf.placeholder(tf.float32, [None, 2, None])
        spec = spec_generator(p, sr)
        audio_float = audio.astype(np.float32) / np.iinfo(np.int16).max
        audio_float = np.expand_dims(audio_float.T, 0)
        s = sess.run(spec, feed_dict={
            p: audio_float,
        })
        f = draw_spectrogram(np.squeeze(s, axis=0), sr)
        plt.show(f)
    '''
    N = tf.shape(frames)[3]
    if remove_dc:
        frames = frames - tf.expand_dims(tf.reduce_mean(frames, axis=3), -1)
    if zero_pad:
        N_fft = tf.to_int32(2**tf.ceil(tf.log(tf.to_double(N)) / np.log(2)))
        print(N_fft)
        frames = tf.pad(frames, [[0, 0], [0, 0], [0, 0], [0, N_fft - N]], "CONSTANT")
    window = get_window(window_type, N, N_t=N_fft) 
    frames = frames * window
    if gpu:
        complex_frames = tf.complex(frames, tf.zeros(tf.shape(frames)))
        spec = tf.fft(complex_frames)
    else:
        def rfft(x):
            return np.fft.rfft(x).astype(np.complex64)
        spec, = tf.py_func(rfft, [frames], [tf.complex64])
    return tf.slice(spec, [0, 0, 0, 0], [-1, -1, -1, N_fft / 2 + 1])

def mel_filterbank(N_fft, sample_rate, num_bands=23, low_freq=0, high_freq=8000):
    '''
    Computes a [N_fft, num_bands] tensor representing the mel filterbank
    Params:
        - N_fft: number of FFT bins
        - sample_rate: the sample rate of the signal
        - num_bands: how many filter bands to use (defaults to 23)
        - low_freq: the lowest frequency in the bands (defaults to 0 Hz, 300 Hz is another good option)
        - high_freq: the highest frequency in the bands (defaults to 8000 Hz)
    '''
    fbank, = tf.py_func(_py_mel_filterbank, [N_fft, sample_rate, num_bands, low_freq, high_freq], [tf.float32])
    fbank.set_shape([None, num_bands])
    return fbank

def apply_filterbank(spec, filter_bank):
    '''
    Params:
      - spec: [batch_size, channels, frames, FFT bins] tensor, the magnitude spectrum
      - filter_bank: [FFT bins, number filters] tensor, the filter bank 
    Returns:
      - [batch_size, channels, frames, number filters] tensor, the filter bank features

    Example:

    import scipy.io.wavfile as wavfile

    def spec_generator(audio, sample_rate):
        frame_length = int(0.025 * sample_rate)
        frame_shift = int(0.015 * sample_rate)
        frames = sliding_window(audio, frame_length, frame_shift, padding="VALID")
        spec = timeseries_to_spec(frames)
        spec = decibels(spec)
        fbank = mel_filterbank(tf.shape(spec)[-1], sample_rate)
        fbank_features = apply_filterbank(spec, fbank)
        return fbank_features

    with tf.Session() as sess:
        sr, audio = wavfile.read("/data/sls/u/urop/kkleidal/three.wav")
        p = tf.placeholder(tf.float32, [None, 2, None])
        fbank = spec_generator(p, sr)
        audio_float = audio.astype(np.float32) / np.iinfo(np.int16).max
        audio_float = np.expand_dims(audio_float.T, 0)
        fbank = sess.run([fbank], feed_dict={
            p: audio_float,
        })
        f = draw_spectrogram(np.squeeze(fbank, axis=0))
        plt.show(f)
    '''
    shape = tf.shape(spec)
    two_d = tf.reshape(spec, [-1, shape[-1]])
    feats = tf.matmul(two_d, filter_bank)
    return tf.reshape(feats, [shape[0], shape[1], shape[2], -1])

###########################
## PRIVATE UTILITY FUNCS ##
###########################

def _freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def _mel_to_freq(mel):
    return 700.0 * (np.power(10, mel / 2595.0) - 1.0)

def _py_mel_filterbank(N_fft, sample_rate, num_bands=23, low_freq=0, high_freq=8000):
    high_freq = min(high_freq, sample_rate)
    mel_bounds = _freq_to_mel(np.array([high_freq, low_freq]))
    mel_points = np.flip(np.linspace(mel_bounds[0], mel_bounds[1], num_bands + 2), 0)
    freq_points = _mel_to_freq(mel_points)
    bin_points = np.floor(float(N_fft + 1) * freq_points / float(sample_rate)).astype(np.int32)
    fbank = np.zeros((N_fft, num_bands))
    for band in range(num_bands):
        start = bin_points[band]
        mid = bin_points[band + 1]
        end = bin_points[band + 2]
        fbank[start:mid+1, band] = np.linspace(0.0, 1.0, mid - start + 1)
        fbank[mid:end+1, band] = np.linspace(1.0, 0.0, end - mid + 1)
    return fbank.astype(np.float32)
