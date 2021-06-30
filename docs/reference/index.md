# API Reference

QAMPy has two different layers of API. The basic API is centred around QAMPy signal objects, an abstraction which contain important information like the sampling and symbol rate, the transmitted bits etc.. All functions in the basic API preserve signal objects. In contrast, functions in the core API typically work on numpy arrays and do not preserve the signal attributes when a signal object is passed to them.

```{toctree}
:maxdepth: 1

basic/index
core/index
```
