from distutils.core import Extension
from setuptools import setup, find_packages, dist#, Extension
#dist.Distribution(dict(setup_requires="pythran"))
# To use a consistent encoding
from codecs import open
from os import path
import sys
import numpy as np
from pythran import PythranExtension

COMPILER_ARGS = ["-O3", "-march=native", "-ffast-math", "-mfpmath=sse", "-funroll-loops", "-fopenmp"]
COMPILER_ARGS_PYT = ["-O3", "-ffast-math", "-mfpmath=sse", "-march=native",
                     "-funroll-loops",
                      "-fopenmp", "-std=c++11", "-fno-math-errno", "-w",
                      "-fvisibility=hidden", "-fno-wrapv", "-DUSE_XSIMD"]
LINK_ARGS = ["-fopenmp", "-lm"]

WIN_COMPILER_ARGS = ["/O2", "/openmp", "/GL", "/fp:fast", "/arch:AVX2"]
WIN_LINK_ARGS = ["/openmp"]

if sys.platform.startswith("win"):
    cython_equalisation = Extension(name="qampy.core.equalisation.cython_equalisation",
                                    sources=["qampy/core/equalisation/cython_equalisation.pyx"],
                                    include_dirs=["qampy/core/equalisation", np.get_include()],
                                    language="c++",
                                    extra_compile_args=WIN_COMPILER_ARGS,
                                    extra_link_args=WIN_LINK_ARGS)
    cython_errorfcts = Extension(name="qampy.core.equalisation.cython_errorfcts",
                                 sources=["qampy/core/equalisation/cython_errorfcts.pyx"],
                                 include_dirs=["qampy/core/equalisation", np.get_include()],
                                 language="c++",
                                 extra_compile_args=WIN_COMPILER_ARGS,
                                 extra_link_args=WIN_LINK_ARGS)
    dsp_cython = Extension(name="qampy.core.dsp_cython",
                           sources=["qampy/core/dsp_cython.pyx"],
                           include_dirs=["qampy/core/equalisation", np.get_include(), "qampy/core/"],
                           language="c++",
                           extra_compile_args=WIN_COMPILER_ARGS,
                           extra_link_args=WIN_LINK_ARGS)
    dsp_pythran = PythranExtension(name="qampy.core.pythran_dsp",
                                   sources = ["qampy/core/pythran_dsp.py"],
                                   extra_compile_args=WIN_COMPILER_ARGS,
                                   extra_link_args=WIN_LINK_ARGS)
    pythran_equalisation = PythranExtension(name="qampy.core.equalisation.pythran_equalisation",
                                            sources = ["qampy/core/equalisation/pythran_equalisation.py"],
                                            extra_compile_args=WIN_COMPILER_ARGS,
                                            extra_link_args=WIN_LINK_ARGS)
else:
    cython_equalisation = Extension(name="qampy.core.equalisation.cython_equalisation",
                                    sources=["qampy/core/equalisation/cython_equalisation.pyx"],
                                    include_dirs=["qampy/core/equalisation", np.get_include()],
                                    extra_compile_args=COMPILER_ARGS,
                                    extra_link_args=LINK_ARGS)
    cython_errorfcts = Extension(name="qampy.core.equalisation.cython_errorfcts",
                                 sources=["qampy/core/equalisation/cython_errorfcts.pyx"],
                                 include_dirs=["qampy/core/equalisation", np.get_include()],
                                 extra_compile_args=COMPILER_ARGS,
                                 extra_link_args=LINK_ARGS)
    dsp_cython = Extension(name="qampy.core.dsp_cython",
                           sources=["qampy/core/dsp_cython.pyx"],
                           include_dirs=["qampy/core/equalisation", np.get_include(), "qampy/core/"],
                           extra_compile_args=COMPILER_ARGS,
                           extra_link_args=LINK_ARGS)
    dsp_pythran = PythranExtension(name="qampy.core.pythran_dsp",
                                   sources = ["qampy/core/pythran_dsp.py"],
                                   extra_compile_args=COMPILER_ARGS_PYT,
                                   extra_link_args=LINK_ARGS)
    pythran_equalisation = PythranExtension(name="qampy.core.equalisation.pythran_equalisation",
                                            sources = ["qampy/core/equalisation/pythran_equalisation.py"],
                                            extra_compile_args=COMPILER_ARGS_PYT,
                                            extra_link_args=LINK_ARGS)



here = path.abspath(path.dirname(__file__))


setup(
    name='qampy',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.2',

    description='A python based package of communications qampy tools',
    long_description=None,

    # The project's main homepage.
    url=None,

    # Author details
    author='Jochen Schröder and Mikael Mazur',
    author_email='jochen.schroeder@chalmers.se',

    # Choose your license
    license='GPLv3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='DSP science',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'test', 'ExampleCode']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'scipy', 'bitarray'],

    #ext_modules = [cython_errorfcts, cython_equalisation, dsp_cython, dsp_pythran, pythran_equalisation],
    ext_modules = [dsp_pythran, pythran_equalisation],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    setup_requires=["pythran"],
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    data_files=[],
)

