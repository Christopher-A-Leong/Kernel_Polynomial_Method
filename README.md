# Kernel Polynomial Method

Code package designed for computing spectral properties of large, sparse Hamiltonians through utilization of the kernel polynomial method (KPM).

## Overview

Computing spectral properties of Hamiltonians of increasing system size quickly becomes unfeasible due to the O(N^3) scaling of exact diagonalization algorithms. One solution to this problem is the kernel polynomial method, which instead computes these properties by reconstructing a Chebyshev polynomial expansion. In this way, the problem becomes one of computing the coefficients (often called moments) of this expansion. Leveraging sparse linear algebra algorithms, KPM is well suited for obtaining insights into the properties of large Hamiltonians (often up to system sizes of 10^9 sites) which would otherwise be impossible to approach with exact diagonalization. In this repository specifically, KPM can be performed in order to compute density of states as well as local density of states of a given Hamiltonian. The former is performed utilizing the well-known stochastic trace evaluation routine. In order to quell Gibbs oscillations, a natural consequence of truncating the Chebyshev polynomial expansion, the Jackson kernel is utilized.

## Requirements

See environment.yml file in the repository to create the conda environment related to this project.

## References

[1] A. Weisse, G. Wellein, A. Alvermann, and H. Fehske, Rev. Mod. Phys. 78, 275 (2006).
