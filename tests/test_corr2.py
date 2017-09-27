import unittest
import numpy as np
import pandas as pd

from scorr import corr2
from scorr._cov_matrices_nofft import xmat2pt_nofft, xmat3pt_nofft

class TestCorrMat(unittest.TestCase):
    def test_simple(self):
        np.testing.assert_array_equal(
            corr2.corr_mat(np.arange(3)),
            np.array([
                [0, 1, 2],
                [2, 0, 1],
                [1, 2, 0]
            ])
        )
        
    def test_maxlag(self):
        np.testing.assert_array_equal(
            corr2.corr_mat(np.arange(5), maxlag=4),
            np.array([
                [0, 1, 2, 3],
                [4, 0, 1, 2],
                [3, 4, 0, 1],
                [2, 3, 4, 0]
            ])
        )
        
    def test_integration(self):     
        x = np.random.normal(size=10)
        y = np.random.normal(size=10)
        C_bruteforce = xmat2pt_nofft(x-np.mean(x), y-np.mean(y), 10)
        C_fft = corr2.corr_mat(corr2.xcorr(x, y, norm='cov', nfft='demix', debias=True), maxlag=10)
        np.testing.assert_almost_equal(C_fft, C_bruteforce)
        
class TestXcorrShift(unittest.TestCase):
    def test_simple(self):
        np.testing.assert_array_equal(
            corr2.xcorrshift([1,2,3]),
            [3,1,2]
        )
        
    def test_maxlag(self):
        np.testing.assert_array_equal(
            corr2.xcorrshift(np.arange(5), maxlag=2),
            [3, 4, 0, 1, 2]
        )
        
    def test_pandas(self):
        pd.testing.assert_series_equal(
            corr2.xcorrshift(pd.Series([1,2,3])),
            corr2.xcorrshift([1,2,3], as_pandas=True)
        )
        
class TestFFTCrop(unittest.TestCase):
    def test_simple(self):
        np.testing.assert_array_equal(
            corr2.fftcrop(np.arange(5), 2),
            [0, 1, 3, 4]
        )
    
class TestXcorr(unittest.TestCase):
    # integration tests in TestAcorr and TestCorrMat!
    def test_xcovariance(self):
        np.testing.assert_array_equal(
            corr2.xcorr(
                [0,1,0,0],[1,0,0,0], 
                nfft=4, norm='cov', subtract_mean=False
            ),
            [0, 0.25, 0, 0]
       )

class TestAcorr(unittest.TestCase):
   def test_randn(self):
       x = corr2.acorr(np.random.normal(size=100))
       np.testing.assert_almost_equal(x[0], 1)
       self.assertTrue(x[1:].mean() < 0.1)
       
   def test_periodic_unpadded(self):
       # test sinus autocorrelation at these times:
       tmax  = 4 * np.pi
       steps = 100
       dt    = tmax / float(steps)
       t     = np.arange(0,tmax,dt)
       # calc autocorrelation directly...
       r_brutforce = []
       for i in range(len(t)):
           r_brutforce.append(np.mean(np.sin(t) * np.sin((t + dt*i))))
       # ...and with acorr
       r_fft = corr2.acorr(np.sin(t), nfft='len', norm='cov', subtract_mean=False)
       np.testing.assert_almost_equal(r_fft, r_brutforce)
        
class TestGroupedAgainstUngrouped(unittest.TestCase):   
       
   def __init__(self, *args, **kwargs):
       # call TestCase __init__
       super(TestGroupedAgainstUngrouped, self).__init__(*args, **kwargs)
       self.steps = 2**10
       self.dlen  = 2**6
       self.x  = np.random.normal(size=self.steps)
       self.y  = np.random.normal(size=self.steps)
       self.df = pd.DataFrame({
           'x': self.x, 
           'y': self.y, 
           'date': (np.arange(self.steps)/self.dlen).astype(int)
       })
   
   def test_acorr_grouped_df(self):
       df  = self.df[['x','date']]
       r   = corr2.acorr(self.x, nfft='pad')
       rdf = corr2.acorr_grouped_df(df, nfft='auto pad')
       self.assertTrue(
           (
               abs(r[:self.dlen/2+1] - rdf['acorr'].loc[0:self.dlen/2]) 
               < 3*rdf['acorr_std'].loc[0:self.dlen/2]
           ).all()
       )
       
   def test_xcorr_grouped_df(self):
       r   = corr2.xcorr(self.x, self.y, nfft='pad')
       rdf = corr2.xcorr_grouped_df(self.df, ['x','y'], nfft='auto pad')
       # positive lags
       self.assertTrue(
           (
               abs(r[:self.dlen/2+1] - rdf['xcorr'].loc[0:self.dlen/2]) 
               < 3*rdf['xcorr_std'].loc[0:self.dlen/2]
           ).all()
       )
       # negative lags
       self.assertTrue(
           (
               abs(r[-self.dlen/2-1:] - rdf['xcorr'].loc[-self.dlen/2:0]) 
               < 3*rdf['xcorr_std'].loc[-self.dlen/2:0]
           ).all()
       )