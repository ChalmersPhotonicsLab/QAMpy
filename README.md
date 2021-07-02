# QAMPy a DSP chain for optical communication signals

[![DOI](https://zenodo.org/badge/124787512.svg)](https://zenodo.org/badge/latestdoi/124787512)

<!-- start description -->
QAMPy is a dsp chain for simulation and equalisation of signals from optical communication transmissions.
It is written in Python, but has been designed for high performance and most performance critical 
functions are written with [pythran](https://github.com/serge-sans-paille/pythran) to run at speed of compiled c or c++
code.

QAMPy can equalise BPSK, QPSK and higher-order QAM signals as well as simulate signal impairments. 

## Equalisation 

For signal equalisation it contains:

 * CMA and modified CMA equalisation algorithms 
 * Radius directed equalisers
 * several decision directed equaliser implementations 
 * phase recovery using blind phase search (BPS) and ViterbiViterbi algorithms
 * frequency offset compensation
 * a complete set of pilot-based equalisation routines, including frame synchronization, frequency offset 
estimation, adaptive equalisation and phase recovery
 * additional data-aided and real-valued adaptive equaliser routines
 
## Impairments
 
It can simulate the following impairments:

 * frequency offset
 * SNR
 * PMD
 * phase noise
 * transceiver impairments such as modulator nonlinearity, DAC frequency response and limited ENOB
 
## Signal Quality Metrics

QAMpy is designed to make working with QAM signals easy and includes calculations for several
performance metrics:

 * Symbol Error Rate (SER)
 * Bit Error Rate (BER)
 * Error Vector Magnitude (EVM)
 * Generalized Mututal Information (GMI)
 
<!-- end description -->

## Documentation

We put a strong focus on documenting our functions and most public functions should be well documented. 
Use help in jupyter notebook to excess the documenation. 

You can access documentation with an extensive API at our [website](http://qampy.org).

For examples of how to use QAMpy see the Scripts and the Notebooks subdirectory, note that not all files are up-to-date
You should in particular look at the *cma_equaliser.py* and *64_qam_equalisation.py* files. 

## Installation
Installation instructions can be found here [here](http://qampy.org/installation/index.html#).

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
