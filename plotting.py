import numpy as np
import matplotlib.pyplot as plt

def draw_spectrogram(spec, sample_rate):
    '''
    Expects spectrogram as [channels, frame count, FFT bins]
    '''
    channels, frame_count, fft_bins = spec.shape
    f, ax = plt.subplots(channels)
    for chan in range(channels):
        s = spec[chan, :, :].T
        ax[chan].imshow(s, interpolation='nearest', aspect='auto', origin='lower')
        ticks = np.arange(0, s.shape[0], s.shape[0] // 8)
        ax[chan].set_yticks(ticks)
        ax[chan].set_yticklabels((sample_rate / (2.0 * s.shape[0]) * ticks).astype(np.int32))
        ax[chan].set_ylabel("Hz")
        ax[chan].set_xlabel("Frame")
        ax[chan].set_title("Channel %d" % (chan + 1))
    return f
