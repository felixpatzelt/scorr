"""Naive direct calculation of correlation matrices"""

import numpy as np
from scorr.corr2 import corr_mat

# naive 2-point covariance matrix for zero-mean signals
def xmat2pt_nofft(x1, x2, maxlag):
    """Return covariance matrix  xmat2pt(x1, x2) = C_x2,x1(lag1,lag2)
    Brute-force, scales like O(n^2)!
    """
    c = np.zeros(2 * maxlag)
    c[0] = np.mean(x1 * x2)
    for l in range(1,maxlag):
        c[-l] = np.mean(x1[:-l] * x2[l:])
        c[l]  = np.mean(x2[:-l] * x1[l:])
    return corr_mat(c, maxlag=maxlag)

# naive 3-point covariance matrix for zero-mean signals
def xmat3pt_nofft(x0, x1, x2, maxlag):
    """Return exact triple covariance (slow, O(n^3)).
    """
    C = np.zeros((maxlag,maxlag))
    for l in range(maxlag):
        for n in range(maxlag):
            if (n > 0) and (l > 0):
                # x0(t) * x1(t+l) * x2(t+n)
                C[l,n] = np.mean(x1[l:-n] * x2[n:-l] * x0[l+n:])
            elif (n > 0):
                C[l,n] = np.mean(x1[:-n] * x2[n:] * x0[l+n:])
            elif (l > 0):
                C[l,n] = np.mean(x1[l:] * x2[:-l] * x0[l+n:])
            elif (n == 0) and (l == 0):
                C[l,n] = np.mean(x2 * x1 * x0)
    return C