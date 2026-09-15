import numpy as np
import scipy
from Lattice.General_Hamiltonian import general_hyperbolic_q3_hamiltonian
from Lattice.General_Hamiltonian_Strain import sublattice_label_q3
from Lattice.General_Hamiltonian_q4 import general_hamiltonian_q4
from Lattice.General_Hamiltonian_q4 import sublattice_label_q4
from Lattice.Honeycomb_Sparse import honeycomb_lattice_sparse
from Lattice.Honeycomb_Sparse import honeycomb_lattice_sparse_PBC
from Lattice.Honeycomb_Sparse import honeycomb_site_assignment


def hamiltonian_hyperbolic_q3_sublattice_basis(pval, nval):
    tbham = general_hyperbolic_q3_hamiltonian(pval, nval)
    asites, bsites = sublattice_label_q3(pval, nval)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    permutation_arr = scipy.sparse.csr_array((np.ones(len(sublattice_basis)),
                                              (np.arange(len(sublattice_basis), dtype=np.int64),
                                               sublattice_basis)),
                                             shape=(tbham.shape[0], tbham.shape[1]))
    return permutation_arr @ tbham @ permutation_arr.T


def hamiltonian_hyperbolic_q4_sublattice_basis(pval, nval):
    tbham = general_hamiltonian_q4(pval, nval)
    asites, bsites = sublattice_label_q4(pval, nval)
    sublattice_basis = np.concat((asites, bsites), dtype=np.int64)
    permutation_arr = scipy.sparse.csr_array((np.ones(len(sublattice_basis)),
                                              (np.arange(len(sublattice_basis), dtype=np.int64),
                                               sublattice_basis)),
                                             shape=(tbham.shape[0], tbham.shape[1]))
    return permutation_arr @ tbham @ permutation_arr.T


def hamiltonian_honeycombOBC_sublattice_basis(nval):
    tbham = honeycomb_lattice_sparse(nval)
    asites, bsites = honeycomb_site_assignment(nval)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    permutation_arr = scipy.sparse.csr_array((np.ones(len(sublattice_basis)),
                                              (np.arange(len(sublattice_basis), dtype=np.int64),
                                               sublattice_basis)),
                                             shape=(tbham.shape[0], tbham.shape[1]))
    return permutation_arr @ tbham @ permutation_arr.T


def hamiltonian_honeycombPBC_sublattice_basis(nval):
    tbham = honeycomb_lattice_sparse_PBC(nval)
    asites, bsites = honeycomb_site_assignment(nval)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    permutation_arr = scipy.sparse.csr_array((np.ones(len(sublattice_basis)),
                                              (np.arange(len(sublattice_basis), dtype=np.int64),
                                               sublattice_basis)),
                                             shape=(tbham.shape[0], tbham.shape[1]))
    return permutation_arr @ tbham @ permutation_arr.T
