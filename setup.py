from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import sys
import numpy as np
from pythran.dist import PythranExtension, PythranBuildExt
import platform

def read(rel_path):
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


COMPILER_ARGS_PYT = ["-O3", "-ffast-math", "-march=native",
                     "-funroll-loops",
                      "-fopenmp", "-std=c++11", "-fno-math-errno", "-w",
                      "-fvisibility=hidden", "-fno-wrapv", "-DUSE_XSIMD",
                     "-DNDEBUG", "-finline-limit=100000"]
# SSE not available on non-x86
if platform.machine() == "x86_64":
    COMPILER_ARGS_PYT += ["-mfpmath=sse"]

LINK_ARGS = ["-fopenmp", "-lm"]
# MacOS Linker does not support stripping symbols
if platform.system() != "Darwin":
    LINK_ARGS += ["-Wl,-strip-all"]

WIN_LINK_ARGS = ["/openmp"]

def make_compile_modules():
    if sys.platform.startswith("win"):
        COMPILER_ARGS_PYT.remove("-DUSE_XSIMD") # windows fails with XSIMD
        dsp_pythran = PythranExtension(name="qampy.core.pythran_dsp",
                                   sources = ["qampy/core/pythran_dsp.py"],
                                   extra_compile_args=COMPILER_ARGS_PYT,
                                   extra_link_args=WIN_LINK_ARGS)
        pythran_equalisation = PythranExtension(name="qampy.core.equalisation.pythran_equalisation",
                                            sources = ["qampy/core/equalisation/pythran_equalisation.py"],
                                            extra_compile_args=COMPILER_ARGS_PYT,
                                            extra_link_args=WIN_LINK_ARGS)
    else:
        dsp_pythran = PythranExtension(name="qampy.core.pythran_dsp",
                                   sources = ["qampy/core/pythran_dsp.py"],
                                   extra_compile_args=COMPILER_ARGS_PYT,
                                   extra_link_args=LINK_ARGS)
        pythran_equalisation = PythranExtension(name="qampy.core.equalisation.pythran_equalisation",
                                            sources = ["qampy/core/equalisation/pythran_equalisation.py"],
                                            extra_compile_args=COMPILER_ARGS_PYT,
                                            extra_link_args=LINK_ARGS)
    return [dsp_pythran, pythran_equalisation]


def check_compile():
    # check if compiling pythran code works if not raise a warning. Modelled after code in pythran toolchain
    from numpy.distutils.core import setup
    from tempfile import mkdtemp, NamedTemporaryFile
    from pythran.dist import PythranExtension, PythranBuildExt
    import warnings
    code = '''
        #include <pythonic/core.hpp>
    '''
    with NamedTemporaryFile(mode="w", suffix=".cpp", delete=False, dir=mkdtemp()) as out:
        out.write(code)
    bdir = mkdtemp()
    btmp = mkdtemp()
    ext = PythranExtension("test", [out.name])
    try:
        setup(name="test",
        packages=[], #this is needed because setuptools otherwise gets confused by the nested setup
        ext_modules=[ext],
        cmdclass={"build_ext": PythranBuildExt},
        script_name='setup.py',
        script_args=['--quiet',
                   'build_ext',
                   '--build-lib', bdir,
                   '--build-temp', btmp]
        )
        return True
    except SystemExit as e:
        warnings.warn("""Test pythran compile failed: {}.
        Do you have a working CLANG++ or GCC compiler installed?
        I will proceed with the installation without compilation. Some functions will be much slower""".format(e))
        return False

#if check_compile():
if True:
    ext_modules = make_compile_modules()
else:
    ext_modules = []

here = path.abspath(path.dirname(__file__))
name = "qampy"
version = get_version("qampy/__init__.py")

setup(
    name=name,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='A python based package of communications qampy tools',
    long_description=None,

    # The project's main homepage.
    url=None,

    # Author details
    author='Jochen Schröder, Mikael Mazur and Zonglong He',
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # What does your project relate to?
    keywords='DSP science',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'test', 'ExampleCode', 'Scripts', 'dist', 'build']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'scipy'],
    ext_modules = ext_modules,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    setup_requires=["pythran>=0.9.7", "sphinx"],
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
    cmdclass = {
        "build_ext": PythranBuildExt, 
    },

)

