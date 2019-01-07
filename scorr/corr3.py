"""Functions to calculate three-point correlations and bispectrum.
"""

import numpy as np
import pandas as pd
from scipy.fftpack import fft, ifft2, fftshift
from scipy.linalg import hankel
try:
    from progress import getLogger
except ImportError:
    from logging import getLogger

from .helpers import is_number_like, is_string_like, get_nfft

def fft2x(x, y, z, nfft=None):
    """Return Bi-cross-spectrum."""
    
    # 1d-fourier-transforms
    # ----------------------------------------------------------
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)
    
    assert len(x) == len(y) and len(x) == len(z), (
        "Arrays must have the same length"
    )
    
    if not nfft:
        # let's hope it's not too much!
        nfft = len(x)
            
    # transform
    xfft = fft(x, n=nfft)
    
    if y is x:
        yfft = xfft
    else:
        yfft = fft(y, n=nfft)

    if z is x:
        zfft = xfft
    elif z is y:
        zfft = yfft
    else:
        zfft = fft(z, n=nfft)
    
    # Fourier space product
    # ----------------------------------------------------------
    ## create indices aligned with fftpack's fft2 quadrants
    lm = nfft / 2.
    i0 = np.roll(
        np.arange(int(np.ceil(lm-1)), int(-lm-1), -1, dtype=int), 
        int(np.floor(lm)) + 1
    )
    i = i0 * np.ones((nfft, nfft), dtype=int)
    j0 = np.arange(0, -nfft, -1, dtype=int)
    j = hankel(j0, np.roll(j0,1))
    
    # B
    xfft = np.conjugate(xfft)
    return xfft[j] * yfft[i] * zfft[i.T]

def padded_x3corr_norm(nfft, pad=0, segments=1, debias=True):
    """Return a matrix of weights necessary to normalise x3corr
    (triple cross-correlation) calculated with zero-padded ffts.
    For pad = zero, all weights are equal to N. See x3corr.
    
    Parameters:
    ===========
    
    nfft: int
        Size of fft segments (= (maximum lag + pad) / 2, see fft)
    pad: int
        Number of zeros added to the segment(s) before fft.
    segments: int
        Number of segments that were summed, is used as multiplyer.
    debias: bool
        Correct for bias from zero padding. Default: False - to be consistent
        with two-point xcorr defaults.
    """
    ndat = nfft - pad
    nmid = min(ndat, nfft//2) # symmetry axis
    nemp = -min(0, ndat-pad) # completely empty
    w3 = np.ones((nfft,nfft))
    
    if (nfft > ndat) and ndat <= pad and debias:
        # correct zero-padding bias
        for n in range(nfft-nemp):
            for k in range(nfft-nemp):
                if n < ndat and k < ndat:
                    w3[n,k] = ndat - max(n,k)
                elif n > ndat and k > ndat:
                    w3[n+nemp,k+nemp] = w3[nfft-n-nemp,nfft-k-nemp]
                elif abs(k - n) > ndat:
                    w3[n+(n>k)*nemp,k+(k>n)*nemp] = max(
                        abs(ndat - abs(k - n)), 1
                    )
    elif (nfft > ndat) and ndat > pad and debias:
        w3 *= ndat - pad
        for n in range(pad):
            for k in range(pad):
                if abs(k-n) < pad:
                    w3[k,n] = w3[-k,n] = w3[k,-n] = w3[-k,-n] = ndat - max(n,k)
    else:
        w3 *= nfft  
          
    return w3 * segments


def x3corr(
        x, y, z, 
        nfft=2**7, pad=0, subtract_mean=True, norm='corr', debias=True
    ):
    """Return triple cross correlation matrix
    
                     < x(t) y(t-k) z(t-l) > 
        C_xyz(k,l) = ----------------------
                      std(x) std(y) std(z)
    
    Parameters:
    ===========
    
    x, y, z: array-like
        Three time-series of the same length
    nfft: int
        Length of fft segments. Output contains nfft/2 positive and
        negative lags.
    pad: int
        Zero-padding of individual fft segments which are averaged.
        default: 0. 
            Result will be nfft x nfft matrix with nfft / 2
            positive and negative lags.
        pad = nfft: 
            Perfect unmixing of positive and negative frequencies.
            Warning: in contrast to the regular cross correlation, 
            parts of the second and third quadrant are be "blind spots"
            where the the distance between the positve- and negative-lag
            contributions exceeds nfft.
    subtract_mean: bool
        Subtract mean from time-series' first.
    norm: str
        default: 'corr' 
            Normalise to correlation
        'cov':
            Normalise to cross covariance
    debias: bool
        Correct for bias from zero padding. Default: True. In contrast to
        the two-point function xcorr, we here debias by default because the
        three-point correlation is always averaged over short segments so
        the bias is significant for all likely applications.
    """
    
    if subtract_mean:
        x = x - np.mean(x)
        y = y - np.mean(y)
        z = z - np.mean(z)
    
    # len of fft segments
    ndat = nfft - pad
    # number of overlapping fft segments (iterations below)
    #nit = int(len(x) / ndat)
    nit = int(np.ceil(len(x) / float(ndat)))
    ti = np.unique(np.linspace(0, len(x)-ndat, nit, dtype=int))
    
    # mean cross-bispectrum
    B = np.zeros((nfft, nfft), dtype=complex)
    for i in range(nit):
        B += fft2x(
            x[ti[i]:ti[i]+ndat], 
            y[ti[i]:ti[i]+ndat], 
            z[ti[i]:ti[i]+ndat],
            nfft = nfft # zero pad
        )
    
    # normalisation
    if norm == "cov":
        n = padded_x3corr_norm(nfft, pad, nit, debias=debias)
    elif is_number_like(norm):
        n = float(norm)
    elif (
            hasattr(norm, '__len__') 
            and len(norm) == len(x) 
            and not is_string_like(norm)
        ):
        n = np.array(norm, dtype=float)
    else:
        n = padded_x3corr_norm(
            nfft, pad, nit * np.std(x) * np.std(y) * np.std(z), debias=debias
        )
    
    # backtransform to t-space & normalise 
    # to obtain the correlation
    return np.real(ifft2(B)) / n
    
def x3corr_grouped_df(
        df, 
        cols,
        by            = 'date', 
        nfft          = 'auto',
        funcs         = (lambda x: x, lambda x: x, lambda x: x), 
        subtract_mean = 'total',
        norm          = 'total corr',
        debias        = True
    ):
    """Group dataframe and calc triple cross correlation for each group 
    separately.
    Returns: mean and std over groups for lags from -nfft/2 to nfft/2.
    
    Parameters:
    ===========
    
    df: pandas.DataFrame
        input time series, must include the columns 
        for which we calculate the xcorr and the one by which we group.
    cols: list of str
        colums with the time series of interest.
    by: str
        column by which to group. default: 'date'
    nfft: str, int
        Length of fft segments. Default: 'auto'.
        'crop':      use the largest power of 2 < smallest group size
        'pad':       use the smallest power of 2 > smallest group size
        'pad > 100': same but ignoring groups with less than 100 events
        'demix':     double-pad to perfectly separate anticausal frequencies.
        Note: 2d-fft can be really inefficient if nfft is not a power of 2.
        See also: get_nfft
    funcs: list of functions
        functions to apply to cols before calculating the xcorr. 
        default: identity (lambda x: x)
    subtract_mean: str
        what to subtract from the time series before calculating the 
        autocorr.
        'total': subtract mean of the whole series from each group
        'group': subtract group mean from each group
        None:    subtract nothing
        default: 'total'
    norm: str
        Normalisation. default: 'total' (normalise normalise days to cov, 
        the end result by total cov giving approx. a correlation.)
        Other Values are passed to xcorr and used on each day separately.
    
    see also: x3corr, xcorr_grouped_df
    """    
    # group, allocate, slice
    g = df.groupby(by)
    # we need three columns
    cols = list(cols)
    assert len(cols) == 3, "Three column-names required."
    df = df[cols]
    g = g[cols]
    # determine nfft
    nfft, events_required = get_nfft(nfft, g)
    
    # what to subtract
    x = None 
    y = None
    z = None
        
    if subtract_mean in ('total', 'auto'):
        # must match normalisation code below!
        x = funcs[0](df[cols[0]])
        y = funcs[1](df[cols[1]])
        z = funcs[2](df[cols[1]])
        
        subtract = [
            x.mean(),
            y.mean(),
            z.mean()
        ]
        sm       = False
    elif subtract_mean in ('group', 'each', True, by):
        subtract = [0,0,0]
        sm       = True
    else:
        subtract = [0,0,0]
        sm       = False
    
    # which norm for each day?
    norm_flag = norm in ("total", "total cov", "total corr", "auto")
    if norm_flag:
        # calculate covariances for each day and later divide by global cov.
        nd = 'cov'
    else:
        nd = norm
    
    # We can't average in frequency space because of the daily normalisation
    ## ( daily variance changes, padding )
    C  = np.zeros((nfft, nfft))
    Ce = np.zeros((nfft, nfft))
    
    # Average over days
    discarded_days = 0
    for i, (gk, gs) in enumerate(g):
        lgs = len(gs)
        if lgs < events_required:
            # this day is too short
            discarded_days += 1
            continue
        else:
            xi  = funcs[0](gs[cols[0]]).values - subtract[0]
            yi  = funcs[1](gs[cols[1]]).values - subtract[1]
            zi  = funcs[2](gs[cols[2]]).values - subtract[1]
            pad = max(nfft - lgs, 0)
            ci  = x3corr(
                xi, yi, zi, 
                nfft=nfft, pad=pad, subtract_mean=sm, norm=nd, debias=debias
            )
            C  += ci
            Ce += ci**2
    
    del ci, xi, yi, zi
    
    n = float(len(g) - discarded_days)
    if discarded_days:
        getLogger(__name__).info(
            "Discarded %i days < %i events" % (
                discarded_days, events_required
            )
        )
        
    if norm_flag:
        if "corr" in norm:
            if x is None:
                # maybe we didn't calculate these yet
                # must match subtract code above!
                x = funcs[0](df[cols[0]])
                y = funcs[1](df[cols[1]])
                z = funcs[2](df[cols[2]])
            # from cross covariance to cross correlation
            n *= np.std(x) * np.std(y) * np.std(z)
        
    C /= n
    Ce = np.sqrt(np.abs(Ce - C**2) / n)
    
    # done
    return C, Ce
    