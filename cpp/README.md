# Faster calculations using C++

Before developing the Monte-Carlo sampling for the `myerson` package the calculation times were improved by rewriting the `restrict` function (called `divide` in the original code) in a compiled language and calling it from Python. 

To make distribution of the Python package easier this code is not implemented in the package.

This code can still be further optimized and will only lead to an actual speedup when compiling with full optimizations. 

Note that to compile this code you need an installation of the boost library.