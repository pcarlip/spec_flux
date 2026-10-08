[![DOI](https://zenodo.org/badge/1353150241.svg)](https://doi.org/10.5281/zenodo.23250619)

# spec_flux
Python tools for calculating spectral flux from fluid velocities. Many currently implicitly assume data from 
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl), though I am working to improve that.

## Installation

Please install either the \[cupy12] or \[cupy13] optional dependency, or otherwise have cupy installed 
(see [the cupy docs](https://docs.cupy.dev/en/stable/install.html)).
I can't include this as an explicit dependency, because the correct version depends on your cuda version, 
but it is necessary to use the package (even without a gpu; the methods to detect gpu use require cupy).
