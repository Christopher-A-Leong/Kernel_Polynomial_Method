import numpy as np
import scipy


def sublattice_label_square(nval):
    if (nval % 2) == 1:
        raise ValueError
    asites = np.array([])
    bsites = np.array([])
    for i in range(nval):
        if (i % 2) == 0:
            asites = np.append(asites, np.arange(i * nval, (i + 1) * nval, 2))
            bsites = np.append(bsites, np.arange(i * nval + 1, (i + 1) * nval, 2))
        else:
            asites = np.append(asites, np.arange(i * nval + 1, (i + 1) * nval, 2))
            bsites = np.append(bsites, np.arange(i * nval, (i + 1) * nval, 2))
    return asites.astype(np.int64), bsites.astype(np.int64)


def square_lattice_sparse_PBC(nval):
    hamiltonian_rows = np.array([])
    hamiltonian_cols = np.array([])

    total_num_sites = int(nval ** 2)
    site_layout = np.arange(0, total_num_sites, dtype=np.int64)
    site_layout = site_layout.reshape(nval, nval)

    # latitudinal bonds
    hamiltonian_rows = np.append(hamiltonian_rows, site_layout.reshape(-1))
    hamiltonian_rows = np.append(hamiltonian_rows, np.roll(site_layout, shift=1, axis=1).reshape(-1))
    hamiltonian_cols = np.append(hamiltonian_cols, np.roll(site_layout, shift=1, axis=1).reshape(-1))
    hamiltonian_cols = np.append(hamiltonian_cols,  site_layout.reshape(-1))

    # longitudinal bonds
    hamiltonian_rows = np.append(hamiltonian_rows, site_layout.reshape(-1))
    hamiltonian_rows = np.append(hamiltonian_rows, np.roll(site_layout, shift=-1, axis=0).reshape(-1))
    hamiltonian_cols = np.append(hamiltonian_cols, np.roll(site_layout, shift=-1, axis=0).reshape(-1))
    hamiltonian_cols = np.append(hamiltonian_cols, site_layout.reshape(-1))

    hamiltonian = scipy.sparse.csr_array((np.ones(len(hamiltonian_rows)),
                                          (hamiltonian_rows.astype(np.int64), hamiltonian_cols.astype(np.int64))),
                                         shape=(total_num_sites, total_num_sites))
    # Correction for counting bonds multiple times for edge case of nval=2
    if nval == 2:
        hamiltonian.data[:] = 1.0

    return hamiltonian
