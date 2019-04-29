# QAMPy a DSP chain for optical communication signals

[![DOI](https://zenodo.org/badge/124787512.svg)](https://zenodo.org/badge/latestdoi/124787512)

QAMPy is a dsp chain for simulation and equalisation of signals from optical communication transmissions.
It is written in Python, but has been designed for high performance and most performance critical 
functions are writen in Cython and run at C-speeds. 

QAMPy can equalise BPSK, QPSK and higher-order QAM signals as well as simulate signal impairments. 

## Equalisation 

For signal equalisation it contains:

 * CMA and modified CMA equalisation algorithms 
 * Radius directed equalisers
 * several decision directed equaliser implementations 
 * phase recovery using blind phase search (BPS) and ViterbiViterbi algorithms
 * frequency offset compensation
 
## Impairments
 
It can simulate the following impairments:

 * frequency offset
 * SNR
 * PMD
 * phase noise
 
## Signal Quality Metrics

QAMpy is designed to make working with QAM signals easy and includes calculations for several
performance metrics:

 * Symbol Error Rate (SER)
 * Bit Error Rate (BER)
 * Error Vector Magnitude (EVM)
 * Generalized Mututal Information (GMI)
 
## Documentation

We put a strong focus on documenting our functions and most public functions should be well documented. 
Use help in jupyter notebook to excess the documenation.

For examples of how to use QAMpy see the Scripts and the Notebooks subdirectory, note that not all files are up-to-date
You should in particular look at the *cma_equaliser.py* and *64_qam_equalisation.py* files. 

## Installation

QAMpy should work for both Python versions 2 and 3, however we mainly test on Python 3, so your mileage might
vary when using Python 2.

QAMPy depends on the following python modules *numpy*, *scipy*, *cython*, *bitarray* and 
*numba* (as a fall-back option). You will also need to have a working c/c++ compiler with open-mp support
installed to compile the cython modules. 

We provide binaries for the latest 0.2 release for Windows and python 3.5-3.7. You can find them under github releases and can 
install them with `pip [filename]`. Note that the builds assume a processor with `sse2` and `avx` extensions, however this should 
be any recent CPU from Intel or AMD. 

## Building 

On Linux we recommend building to get the best performance, see the instructions below. Building on Windows is also possible 
but typically a bit more complicated.

### Linux

On Linux installation works fine using the usual `python3 setup.py build` and `python3 setup.py install`.

### Windows

On Windows, cython modules need to be compiled with the same compiler as python was compiled with, which 
is often a very old version of Visual Studio. If you do not use the correct Visual Studio version
you will often encounter cryptic compile errors. We will try to include more specific instructions in the future. 

## Status

QAMpy is still in alpha status, however we daily in our work. We will try to keep the basic API stable
across releases, but implementation details under core might change without notice.

## Licence and Authors

QAMpy was written by Mikael Mazur and Jochen Schröder from the Photonics Laboratory at Chalmers University of Technology 
and is licenced under GPLv3 or later. 

## Citing

If you use QAMpy in your work please cite us as `Jochen Schröder and Mikael Mazur, "QAMPy a DSP chain for optical 
communications, DOI: 10.5281/zenodo.1195720"`.

## Acknowledgements
The GPU graphics card used for part of this work was donated by NVIDIA Corporation
