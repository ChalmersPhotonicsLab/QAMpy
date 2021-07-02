# Windows installation

Generally python package installation is more complicated on Windows, we therefore recommend to use the (Anaconda)[https://www.anaconda.com/products/individual] or (Miniconda)[https://docs.conda.io/en/latest/miniconda.html] distributions. The instructions below assume that you are using either of these.

## Installation of dependencies

1. Open an anaconda console and install numpy and scipy using `conda install numpy`, `conda install scipy`. Alternatively you can also use the anaconda navigator for installation.
```{note} If you are using the miniconda distribution you likely want to install other packages like matplotlib and jupyter
```
2. Install pythran with `conda install -c conda-forge pythran`
3. The previosly needed `pythran-openblas` package is no longer needed.


## Prebuild packages

Because building on  windows is often somewhat more complicated, we provide 
binaries for the latest 0.3 release for Windows and python 3.5-3.8. You can find them under github releases and can 
install them with `pip [filename]`. However, for best performance building yourself is still recommended.

```{note}
Note that the builds assume a processor with `sse2` and `avx` extensions, however this should 
be any recent CPU from Intel or AMD.  
```


## Building

To build QAMPy on windows you will need:
1. The Microsoft developer tools from the visual studio community edition which can be found [here](https://visualstudio.microsoft.com/vs/community/). You do need
to only install the Desktop developmentwith C++ and the Python development  packages. 
2. clang-cl.exe from the [llvm](https://llvm.org/) project. The compiler can be installed from the binary packages provided by the project, however we recommend to install using conda by running `conda install -c conda-forge clang` in an anaconda console.

Once the dependencies have been installed, open a anaconda console, navigate to the QAMPy directory and build and install with with  `python setup.py build` and `python setup.py install`






