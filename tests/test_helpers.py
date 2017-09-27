import unittest
import numpy as np
import pandas as pd
import scorr.helpers as sh

class TestIsStringLike(unittest.TestCase):
    def test_str(self):
        self.assertTrue(sh.is_string_like('foo'))
    def test_num(self):
        self.assertFalse(sh.is_string_like(42))

class TestIsNumberLike(unittest.TestCase):
    def test_str(self):
        self.assertFalse(sh.is_number_like('foo'))
    def test_num(self):
        self.assertTrue(sh.is_number_like(42))
    def test_array_0D(self):
        self.assertTrue(sh.is_number_like(np.array([1])))
    def test_array_1D(self):
        self.assertFalse(sh.is_number_like(np.array([1,2])))

class TestGetNfft(unittest.TestCase):
    # test data
    def __init__(self, *args, **kwargs):
        # call TestCase __init__
        super(TestGetNfft, self).__init__(*args, **kwargs)
        self.df = pd.DataFrame({
            'x': np.zeros(20), 'd': (np.arange(20)/5).astype(int)
        })
        self.dfg = self.df.groupby('d')
        
    # testers
    def test_passthrough(self):
        self.assertEqual(sh.get_nfft(128, self.dfg), (128, 64))
    def test_pad(self):
        self.assertEqual(sh.get_nfft('pad', self.dfg), (8, 5))
    def test_pad_lg_error(self):
        with self.assertRaises(AssertionError):
            sh.get_nfft('pad > 6', self.dfg)
    def test_demix(self):
        self.assertEqual(sh.get_nfft('demix', self.dfg), (16, 5))
    def test_crop(self):
        self.assertEqual(sh.get_nfft('crop', self.dfg), (4, 5))
    def test_len(self):
        self.assertEqual(sh.get_nfft('len', self.dfg), (5, 5))
