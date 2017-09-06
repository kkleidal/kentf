import numpy as np
import matplotlib.pyplot as plt

def plot_spec(spec, ax=None, sample_rate=None, cmap="inferno_r", title="Spectrogram"):
    '''
    Plot a 1 channel spectrogram as a heatmap in matplotlib
     - ax is the axis on which to plot the heatmap
     - a color series is returned for the user to use to plot a colorbar.
     - if no axis is provided, the current axis is used by default and a colorbar is added
       with plt.colorbar
    '''
    if ax is None:
        pltax = plt.gca()
    else:
        pltax = ax
    spec = np.transpose(spec, [1, 0])
    c = pltax.imshow(spec, cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')
    if sample_rate is not None:
        newlabels = (pltax.get_yticks() / float(2 * (spec.shape[0] - 1)) * sample_rate).tolist()
        pltax.set_yticklabels(newlabels)
        pltax.set_ylabel("Frequency (Hz)")
    else:
        pltax.set_ylabel("FFT Bin")
    pltax.set_xlabel("Frame Number")
    pltax.set_title(title)
    if ax is None:
        plt.colorbar(c)
    return c
