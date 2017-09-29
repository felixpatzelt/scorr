from setuptools import setup, find_packages

setup(
    name='scorr',
    version='1.0.0',
    description=(
        'Fast and flexible two- and three-point correlation analysis '
        'for time series using spectral methods.'
    ),
    long_description="""
        FFT-based correlation analysis for optimal performance. Offers many 
        options for normalisation, mean removal, averaging, zero-padding and
        demixing, as well as bias correction. 
        In particular, averaging over pandas groups of different sizes 
        (e.g. different days) is supported.
    """,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2 :: Only',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords=[
        'two-point', 'three-point', 'cross-correlation', 'autocorrelation', 
        'bispectrum', 'time series', 'pandas', 'correlation matrix'
    ],
    url='http://github.com/felixpatzelt/scorr',
    download_url=(
      'https://github.com/felixpatzelt/scorr/archive/1.0.0.tar.gz'
    ),
    author='Felix Patzelt',
    author_email='felix@neuro.uni-bremen.de',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas'
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite='tests',
)