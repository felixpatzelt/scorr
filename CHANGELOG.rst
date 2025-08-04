Changelog
=========

:Version: 1.1.0 of 2025-08-04

Remove the undocumented __reload_submodules__ function. Reasons:
- It was only ever used for debugging, better tools exist now
- It relied on `imp`, which was removed in Python 3.12


:Version: 1.0.1 of 2019-01-07

Add Python 3 support.


:Version: 1.0.0 of 2017-09-29

Some minor documentation and metadata changes. This version is included with 
Patzelt and Bouchaud (JSTAT 2017) as an online supplement.


:Version: 1.0-rc.1 of 2017-09-25

Refactoring into the scorr module for public release in pip-installable form.


:Version: [unreleased] of 2009-03-06

The functions provided by scorr were part of a large personal tool collection 
for research. Development started after switching to Python and missing a 
*fast* equivalent to MATLAB's xcorr. Functionality was added over time as 
needed, finally three-point correlations in 2017.
