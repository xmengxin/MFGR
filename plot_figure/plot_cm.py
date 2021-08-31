
"""Functions for plotting confusion matrices."""
from __future__ import print_function
from __future__ import unicode_literals
import warnings

from matplotlib import pyplot as pyp
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import cm
from matplotlib.colors import LogNorm
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore', message="Singular matrix")
warnings.filterwarnings('ignore', message="elementwise comparison failed")


ASPECT = 10. # Ratio of x-axis to y-axis.
EPS = 1e-5


def plot_cm(y_true, y_pred, labels, norm_axis=1, cmap='jet', vmin=0,
            vmax=None, lognorm=False, xlabel='True class',
            ylabel='Predicted class', title=None):
    """Plot confusion matrix.
    Parameters
    ----------
    y_true : ndarray, (n_samples,)
        True classes.
    y_pred : ndarray, (n_samples,)
        Predicted classes.
    labels : list of str
        List of class labels, in order that they should be displayed in
        confusion matrix.
    norm_axis : int, optional
        Normalize confusion matrix to sum to one along specified axis. Relevant for text plotted inside cells.
        (Default: 1)
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap to use.
        (Default: 'Reds')
    vmin : int, optional
        Lower unnormalized count cutoff for colormap.
        (Default: 0)
    vmax : int, optional
        Upper unnormalized count cutoff for colormap. If None, then inferred
        from data.
        (Default: None)
    lognorm : bool, optional
       If True, use log-scale for colormap.
       (Default: False)
    xlabel : str, optional
        Label for x-axis.
        (Default: 'True class')
    ylabel : str, optional
        Label for y-axis.
        (Default: 'Predicted class')
    title : str, optional
        Title for plot.
        (Default: None)
    Returns
    -------
    ax : matplotlib.pyplot.Axes
        Axes.
    sigma : ndarray, (n_classes, n_classes)
        Unnormalized confusion matrix. ``sigma[i, j]`` is the number of times
        true label ``labels[i]`` was predicted to be label ``labels[j]``.
    sigma_norm : ndarray, (n_classes, n_classes)
        Normalized confusion matrix. Equivalent to ``sigma`` normalized so that
        rows (``norm_axis=1``) or columns (``norm_axis=0``) sum to 1.
    """
    cmap = cm.jet if cmap == 'jet' else cmap # Protect against seaborn's hate
                                             # for jet.

    # Compute confusion matrix.
    sigma = confusion_matrix(y_true, y_pred, labels=labels)
    if norm_axis in [0, 1]:
        marginals = sigma.sum(axis=norm_axis, dtype='float64')
        sigma_norm = sigma / np.expand_dims(marginals, axis=norm_axis)
    else:
        raise ValueError('norm_axis must be one of {0, 1}. Got %d' % norm_axis)
    sigma = sigma + EPS
    sigma_norm = sigma_norm + EPS

    # Plot raw counts.
    vmin = max(vmin, EPS)
    kwargs = {'vmin' : max(vmin, EPS),
              'vmax' : vmax,
              }
    if lognorm:
        kwargs['norm'] = LogNorm()
    ax = sns.heatmap(sigma, xticklabels=labels, yticklabels=labels,
                     cmap=cmap, robust=True, square=False, cbar=True)
    ### show labels on x and y axis
    xticks = ax.xaxis.get_major_ticks()
    for i in range(len(xticks)):
        if i == 0:
            xticks[i].set_visible(True)
        elif (i+1) % 10 == 0:
            xticks[i].set_visible(True)
        else:
            xticks[i].set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    for i in range(len(yticks)):
        if i == 0:
            yticks[i].set_visible(True)
        elif (i+1) % 10 == 0:
            yticks[i].set_visible(True)
        else:
            yticks[i].set_visible(False)
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    if title is not None:
        ax.set_title(title)
    pyp.tight_layout()
    return ax, sigma, sigma_norm


if __name__ == '__main__':
    labels = np.array(list(range(1,61)))
    n_classes = len(labels)
    y_true = labels[np.random.randint(n_classes, size=1000)]
    y_pred = labels[np.random.randint(n_classes, size=1000)]
    pyp.figure()
    ax, sigma, sigma_norm = plot_cm(y_true, y_pred, labels,
                                    vmax=50, title='Confusion matrix')
    pyp.savefig('test1.jpg')
    pyp.show()