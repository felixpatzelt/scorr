scorr
=====

Fast two- and three-point correlation analysis for time series
using spectral methods.

The methods are FFT-based for optimal performance and offer many options for 
normalisation, mean removal, averaging, and zero-padding. In particular, 
averaging over pandas groups of different sizes (e.g. different days) is 
supported.

The algorithms to calculate three-point correlations described in:
	
    Patzelt, F. and Bouchaud, J-P.:
    Nonlinear price impact from linear models
    Journal of Statistical Mechanics (2017)
    Preprint at arXiv:1706.04163
    

Installation
------------

	pip install scorr
	
		
Dependencies
------------

	- Python 2.7
	- NumPy
	- SciPy
	- Pandas
