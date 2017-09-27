import unittest
import numpy as np
import pandas as pd

from scorr import corr3
from scorr._cov_matrices_nofft import xmat2pt_nofft, xmat3pt_nofft

# Here we only do integration testing since fft2x and padded_xcorr_norm
# are always used anyway. "Wichtig ist was hintern raus kommt" -Helmut K.

class TestXcorr3(unittest.TestCase):
    def test_convergence_against_bruteforce(self): 
        # this is a rather weak test, but so be it
        C_err0  = 0.1
        samples = 64, 128, 256
        for s in samples:
            t = np.arange(s) / 2 / np.pi
            x = np.sin(t/1.5)
            y = np.sin(t/2.)
            z = np.cos(t)
            C_bruteforce = xmat3pt_nofft(
                x-np.mean(x), y-np.mean(y), z - np.mean(z), 8
            )
            C_fft = corr3.x3corr(   
                x, y, z, norm='cov', nfft=32, pad=16
            )
            C_err =  np.abs(C_bruteforce[:8,:8] - C_fft[:8,:8]).max()
            self.assertTrue(C_err < C_err0)
            C_err0 = C_err
            
    def test_corr_invariance(self):
        t = np.arange(64) / 2 / np.pi
        x = np.sin(t/1.5)
        y = np.sin(t/2.)
        z = np.cos(t)
        C1 = corr3.x3corr(   
            x, y, z, norm='corr', nfft=32, pad=16
        )
        C2 = corr3.x3corr(   
            2*x, 2*y, 2*z, norm='corr', nfft=32, pad=16
        )
        np.testing.assert_almost_equal(C1, C2)

class TestGroupedAgainstUngrouped(unittest.TestCase):   
    def __init__(self, *args, **kwargs):
    # call TestCase __init__ 
        super(TestGroupedAgainstUngrouped, self).__init__(*args, **kwargs)
        t = np.arange(128)
        dlen = 16
        self.x = np.random.randn(len(t))
        self.y = np.random.randn(len(t))
        self.z = np.random.randn(len(t))
        self.df = pd.DataFrame({
            'x': self.x, 
            'y': self.y, 
            'z': self.z, 
            'date': (t/dlen).astype(int)
        })
        
    # len, crop, and pad are equal for input lengths that are powers of two
    # the correct choice of input lengths should be well-tested in the
    # get_nnft unittest.
    def test_x3corr_grouped_df_len(self):
        # test without padding
        C = corr3.x3corr(   
            self.x, self.y, self.z, nfft=16, pad=0
        )
        Cdf, Cdfe = corr3.x3corr_grouped_df(self.df, ['x','y','z'], nfft='len')
        self.assertTrue(((abs(C) != 0) * (abs(Cdf - C)  / Cdfe)).max() < 2)
        
    def test_x3corr_grouped_df_demix(self):
        # test demixing - more challenging due to boundary effects
        C = corr3.x3corr(   
            self.x, self.y, self.z, nfft=32, pad=16
        )
        Cdf, Cdfe = corr3.x3corr_grouped_df(self.df, ['x','y','z'], nfft='demix')
        self.assertTrue(((abs(C) != 0) * (abs(Cdf - C)  / Cdfe)).max() < 3.5)
        