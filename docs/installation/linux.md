# Linux installation

The performance critical code in QAMPy is compiled using the [pythran](https://github.com/serge-sans-paille/pythran) ahead of time compiler. 
Installation therefore requires compilation of the pythran generated code. 
On Linux do not provide prebuild packages for linux, instead we recommend to build the packages yourself as this quite straight-forward 
and yields significantly better performance

## Install dependencies

1. Install `scipy, numpy, pythran` using your package manager. 
2. Install a C++ compiler (clang++ or gcc both work) and open-mp libraries to take advantage of multi-core processors, on Ubuntu you need to install
the `libgomp-dev` package if your using gcc or the `libomp-dev` package for llvm(clang). On other distributions packages have similar names. 
3. Previously QAMPy required the openblas package for faster dot product calculation, however recent versions do not link to openblas. If you are going to use
pythran to speed up other code it is still highly recommend to install openblas however.

```{note} You can modify the pythran configuration by creating a $HOME/.pythranrc file please see the [pythran-documentation](https://pythran.readthedocs.io/en/latest/MANUAL.html) for details. However, compiler flags are defined in setup.py, and are optimised for QAMPy compilation.
```

## Building

Once the requirements are installed QAMPy building and installation can be done with  `python setup.py build` and `python setup.py install`


