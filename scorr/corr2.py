"""Functions to calculate two-point correlations.
"""

import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft
from scipy.linalg import toeplitz
try:
    from progress import getLogger
except ImportError:
    from logging import getLogger

from .helpers import is_number_like, is_string_like, get_nfft
    
# Helpers
# ===========================================================================
def corr_mat(x, maxlag=None):
    """Return correlation matrix from correlation array.
    
    Parameters:
    ===========
        x: array-like
            Correlation array in the form returned by e.g. acorr, xcorr.
            NOT centered!
        maxlag: int
            Maximum lag to consider (should be < len(x) / 2).
    """
    # | c_0  c_1 ... c_L |
    # | c_-1 c_0 ...     |
    # | ...              |
    # | c_-L ...     c_0 |
    if maxlag:
        # topeliz(
        #   first_column(l=0,-1,-2,...,-maxlag), first_row(l=0,1,2,...,+maxlag)
        # )
        return toeplitz(np.concatenate([[x[0]], x[:-maxlag:-1]]), x[:maxlag])
    else:
        return toeplitz(np.concatenate([[x[0]], x[:0:-1]]), x)


def xcorrshift(x, maxlag=None, as_pandas=False):
    """Return shifted (cross- / auto) correlation to center lag zero."""
    if not maxlag:
        maxlag = len(x) // 2
    # force pandas output?
    if as_pandas and not hasattr(x, 'iloc'):
        if len(np.shape(x)) > 1:
            x = pd.DataFrame(x)
        else:
            x = pd.Series(x)
    # slice
    ix = np.arange(-maxlag, maxlag+1, dtype=int)
    if hasattr(x, 'iloc'):
        xs = x.iloc[ix]
        xs.index = ix
    else:
        try:
            xs = x[ix]
        except:
            xs = np.asanyarray(x)[ix]
    return xs

def fftcrop(x, maxlag):
    """Return cropped fft or correlation (standard form starting at lag 0)."""
    return np.concatenate([x[:maxlag], x[-maxlag:]])
    
    
def padded_xcorr_norm(nfft, pad, debias=False):
    """Return a vector of weights necessary to normalise xcorr
    (cross-correlation) calculated with zero-padded ffts.
    For pad = 0, all weights are equal to N.
    
    Parameters:
    ===========
    
    nfft: int
        Length of the fft segment(s)
    pad: int
        Number of padded zeros
    """
    ndat = nfft - pad
    if pad <= 0:
        w = nfft * np.ones(1)
    elif debias:
        nmp = max(1, ndat - pad)
        w = np.concatenate([
            np.arange(ndat,nmp, -1), # lag0, lag1, ...
            nmp * np.ones(max(0, nfft - 2 * (ndat-nmp)+1)), # lags > ndat
            np.arange(nmp+1, ndat,1)  # ...lag-1
        ])
    else:
        w = ndat * np.ones(1)
    return w

# For arrays
# ===========================================================================

def xcorr(
        x, y, 
        norm='corr', 
        nfft='auto', 
        subtract_mean=True, 
        debias=False, 
        e=0
    ):
    """Return cross-correlation or covariance calculated using FFT.
    
    Parameters:
    -----------
    
    x, y: array-like (1-D)
        Time series to analyse.
    norm: [optional]
        How to normalise the result
            "corr": Return correlation, i.e. r \\in [-1, 1] (default).
            "cov":  Return covariance. E.g. the peak of an autocorrelation 
                    will have the height var(x) = var(y)
            int, float:
                    Normalise result by this number.
    nfft: int, str [optional]
        How to set the length of the FFT (default: 'pad').
        'len':    Always use len(x), exact for periodic x, y.
        'pad':    Pad length to next number of two.
        'demix':  Zero-pad to demix causal and anti-causal part, giving
                  the exact result for an aperiodic signal.
        'auto':   Equal to 'len' for short series and 'pad' for long series
                  for better performance. This setting is appropriate when
                  the maximum lag of interest much smaller then half the signal
                  length.
        int:      Passed through to fft.
    subtract_mean: bool [optional]
        Subtract the signals' means (default: True).
    debias: bool [optional]
        True:  Correct the bias from zero-padding if applicable.
               This corresponds to the assumption that x, y are segments
               of two stationary processes.
               The SNR will decrease with |lag| because the number of
               data points decreases. 
        False: Don't correct. This corresponds to the assumption that x and y
               are zero outside of the observed range. As a consequence,
               the correlation (or covariance) converges to zero for long lags.
        Default: False because the bias is only significant compared to the
        noise level when many short segments are averaged. It is also
        consistent with similar functions like e.g. numpy.correlate.
    e: float [optional]
        Small epsilon to add to normalisation. This avoids e.g. blowing
        up correlations when the variances of x, y are extremely small.
        Default: 0.
            
    Notes:
    -----
    
    The Fourier transform relies on the assumption that x and y are periodic.
    This may create unexpected resuls for long lags in time series that are
    shorter than the correlation length. To mitigate this effect, consider 
    nfft='pad'.
    
    The output is uncentered, use xcorrshift to center.
    
    The parameter combination 
        nfft='pad', norm=1, subtract_mean=False, debias=False
    corresponds to numpy.correlate with mode='full'.
    """
    
    lx = len(x)
    assert lx == len(y), "Arrays must have the same length"
    
    # padding for demixing and higher performance
    crop_pad = False
    if nfft == 'auto':
        if lx >= 10**4:
            nfft = 'pad'
        else:
            nfft = 'len'
    if nfft == 'demix':
        nfft = int(2**(np.ceil(np.log2(len(x))) + 1))
        crop_pad = True
    elif nfft == 'pad':
        nfft = int(2**(np.ceil(np.log2(len(x)))))
        crop_pad = True
    elif nfft == 'len':
        nfft = lx
    else:
        assert nfft == int(nfft), "nfft must be either 'pad', 'len', or an int"
    #print "xcorr nfft:", nfft
    
    # flatten arrays to 1 dimension, extracts values from pd.Dataframe too
    x = np.ravel(x)
    y = np.ravel(y)
    
    # fourier transform of x
    if subtract_mean:
        # normally the mean is subtracted from the signal
        x = x-np.mean(x)
    
    xfft = fft(x, n=nfft)
    
    # fourier transform of y
    if x is y:
        yfft = xfft
    else:
        if subtract_mean:
            y = y-np.mean(y)

        yfft = fft(y, n=nfft)
    
    # inverse transform
    r = np.real(ifft(xfft * np.conjugate(yfft)))
    
    del xfft, yfft
    
    # normalisation
    ly = padded_xcorr_norm(nfft, nfft - len(y), debias=debias)
    if norm == "cov":
        n = ly
    elif is_number_like(norm):
        n = np.asanyarray(norm, dtype=float)
    else:
        n = ly
        if x is y:
            n *= np.var(x)
        else:
            n *= np.std(x) * np.std(y)
    # done
    r =  r / (n + e)
    if crop_pad:
        r = fftcrop(r, lx)
    return r

    
def acorr(y, **kwargs):
    """Return autocorrelation, equivalent to xcorr(y,y, **kwargs).
    See xcorr for documentation.
    """
    r = xcorr(y, y, **kwargs)
    return r


# For pandas
# ===========================================================================

def xcorr_grouped_df(
        df, 
        cols,
        by            = 'date', 
        nfft          = 'pad', 
        funcs         = (lambda x: x, lambda x: x), 
        subtract_mean = 'total',
        norm          = 'total',
        return_df     = True,
        debias        = True,
        **kwargs
    ):
    """Group dataframe and calc cross correlation for each group separately.
    Returns: mean and std over groups.
    
    Parameters:
    ===========
    
    df: pandas.DataFrame
        input time series, must include the columns 
        for which we calculate the xcorr and the one by which we group.
    cols: list of str
        colums with the time series' of interest.
    by: str [optional]
        column by which to group. default: 'date'
    nfft: int, str [optional]
        Twice the maximal lag measured. default: 'pad'
        'len':        use smallest group size. 
        'pad > 100':  zero pad to next power of two of smallest froup size
                      larger than 100. I.e. at least 128.
        ... see get_nfft for more details
    funcs: list of functions [optional]
        functions to apply to cols before calculating the xcorr. 
        default: identity (lambda x: x)
    subtract_mean: str [optional]
        what to subtract from the time series before calculating the 
        autocorr.
        'total': subtract mean of the whole series from each group
        'group': subtract group mean from each group
        None:    subtract nothing
        default: 'total'
    norm: str [optional]
        Normalisation. default: 'total' (normalise normalise days to cov, 
        the end result by total cov giving approx. a correlation.)
        Other Values are passed to xcorr and used on each day separately.
    return_df: bool
        Return a pandas.DataFrame. Default: True.
    debias: bool [optional]
        True:  Correct the bias from zero-padding if applicable (default).
        False: Don't debias.
            
    **kwargs are passed through. see also: acorr, xcorr, acorr_grouped_df
    """
    # group, allocate, slice
    g = df.groupby(by)
    
    # we always need columns
    cols = list(cols)
    df = df[np.unique(cols)]
    g = g[cols]
    
    # determine fft segment size        
    nfft, events_required = get_nfft(nfft, g)
    maxlag = int(min(nfft//2, events_required))
    
    # allocate
    acd = np.zeros((2*maxlag, len(g)))
    
    # what to subtract
    fdf0 = None
    fdf1 = None
    if subtract_mean in ('total', 'auto'):
        # must match normalisation code below
        fdf0 = funcs[0](df[cols[0]])
        fdf1 = funcs[1](df[cols[1]])
        subtract = [
            fdf0.mean(),
            fdf1.mean(),
        ]
        sm       = False
    elif subtract_mean in ('group', 'each', True, by):
        subtract = [0,0]
        sm       = True
    else:
        subtract = [0,0]
        sm       = False
    
    # which norm for each day?
    if norm in ("total", "auto"):
        # calculate covariances for each day and later divide by global cov.
        nd = 'cov'
    else:
        nd = norm
        
    # do it
    discarded_days = 0
    for i, (gk, gs) in enumerate(g):
        if len(gs) < events_required:
            # this day is too short
            discarded_days += 1
            continue
        else:
            x = np.zeros(nfft)
            # average over minimally overlapping segments
            nit = int(np.ceil(len(gs) / float(nfft)))
            tj = np.unique(np.linspace(0, len(gs)-nfft, nit, dtype=int))
            for j in range(nit):
                x += xcorr(
                    funcs[0](gs[cols[0]][tj[j]:tj[j]+nfft]) - subtract[0], 
                    funcs[1](gs[cols[1]][tj[j]:tj[j]+nfft]) - subtract[1], 
                    subtract_mean=sm,
                    norm   = nd,
                    nfft   = nfft,
                    debias = debias,
                    **kwargs
                )
            acd[:,i] = fftcrop(x / nit, maxlag)
            del x
    
    # average
    acdm = acd.mean(axis=1)
    acde = acd.std(axis=1)
    
    n = 1.
    if norm in ("total", "auto"):
        if fdf0 is None:
            # maybe we didn't calculate these yet
            # must match subtract code above!
            fdf0 = funcs[0](df[cols[0]])
            fdf1 = funcs[1](df[cols[1]])
        # from cross covariance to cross correlation
        n = 1./(np.std(fdf0) * np.std(fdf1))
        
    if discarded_days:
        getLogger(__name__).info(
            "Discarded %i %ss < %i events" % (
                discarded_days, by, events_required
            )
        )
        n *= len(g) / float(len(g) - discarded_days)
    
    acdm *= n
    acde *= n
    
    # done
    if return_df:
        lag = pd.Index(list(range(-maxlag,maxlag+1)), name='lag')
        return pd.DataFrame({
                'xcorr':     xcorrshift(acdm, maxlag),
                'xcorr_std': xcorrshift(acde, maxlag),
        }, index=lag)
    else:
        return acdm, acde
    
def acorr_grouped_df(
        df, 
        col           = None, 
        by            = 'date', 
        nfft          = 'pad', 
        func          = lambda x: x, 
        subtract_mean = 'total',
        norm          = 'total',
        return_df     = True,
        debias        = True,
        **kwargs
    ):
    """Group dataframe and calc autocorrelation for each group separately.
    Returns: mean and std over groups for positive lags only.
    
    Parameters:
    ===========
    
    df: pandas.DataFrame, pandas.Series
        input time series. If by is a string, df must include the column 
        for which we calculate the autocorr and the one by which we group.
        If by is a series, df can be a series, too.
    col: str, None [optional]
        column with the time series of interest.
    by: str [optional]
        column by which to group. default: 'date'
    nfft: int, str [optional]
        twice the maximal lag measured. default: 'auto'
        'auto': use smallest group size. 
        'auto pad > 100':   zero pad to segments of length >= 200,
                            skip days with fewer events
    func: function [optional]
        function to apply to col before calculating the autocorr. 
        default: identity.
    subtract_mean: str [optional]
        what to subtract from the time series before calculating the 
        autocorr.
        'total': subtract mean of the whole series from each group
        'group': subtract group mean from each group
        None:    subtract nothing
        default: 'total'
    norm: str [optional]
        default: 'total' (normalise mean response to one at lag zero).
        Other values 
    debias: bool [optional]
        True:  Correct the bias from zero-padding if applicable (default).
        False: Don't debias.
    
    **kwargs are passed through. see also: acorr, xcorr, xcorr_grouped_df
    """    
    # group, allocate, slice
    g = df.groupby(by)
    
    if not col:
        if (
                    is_string_like(by) 
                and hasattr(df, 'columns') 
                and by in df.columns
            ):
            # we just got two columns, one is group, so it's clear what to do
            col = list(df.columns)
            col.remove(by)
        elif len(df.shape) > 1:
            # unclear what to do
            raise ValueError
    
    # determine fft segment size
    nfft, events_required = get_nfft(nfft, g)
    maxlag = int(min(nfft//2, events_required))
    
    # allocate
    acd = np.zeros((maxlag + 1, len(g)))
    
    # what to subtract
    fdf = None
    if subtract_mean in ('total', 'auto'):
        subtract = func(df[col]).mean()
        sm = False
    elif subtract_mean in ('group', 'each', True, by):
        subtract = 0
        sm = True
    else:
        subtract = 0
        sm = False
        
    # which norm for each day?
    if norm in ("total", "auto"):
        # calculate covariances for each day, later norm to one giving a corr.
        nd = 'cov'
    else:
        nd = norm
        
    # do it
    discarded_days = 0
    for i, (gk, gs) in enumerate(g):
        if len(gs) < events_required:
            # this day is too short
            discarded_days += 1
            continue
        else:
            x = np.zeros(maxlag+1)
            # average over minimally overlapping segments
            nit = int(np.ceil(len(gs) / float(nfft)))
            tj = np.unique(np.linspace(0, len(gs)-nfft, nit, dtype=int))
            for j in range(nit):
                x += acorr(
                    func(gs[col][tj[j]:tj[j]+nfft]) - subtract, 
                    subtract_mean=sm,
                    norm   = nd,
                    nfft   = nfft,
                    debias = debias,
                    **kwargs
                )[:maxlag+1]
            acd[:,i] = x / nit
            del x
    
    # average
    acdm = acd.mean(axis=1)
    acde = acd.std(axis=1)
    
    n = 1
    if norm in ("total", "auto"):
        # norm to one
        n = 1./acdm[0]
    elif discarded_days:
        n = len(g) / float(len(g) - discarded_days)
        
    if discarded_days:
        getLogger(__name__).info(
            "Discarded %i %ss < %i events" % (
                discarded_days, by, events_required
            )
        )
    
    acdm *= n
    acde *= n
    
    # done
    if return_df:
        lag = pd.Index(list(range(maxlag+1)), name='lag')
        return pd.DataFrame({
                'acorr':     acdm,
                'acorr_std': acde,
        }, index=lag)
    else:
        return acdm, acde

