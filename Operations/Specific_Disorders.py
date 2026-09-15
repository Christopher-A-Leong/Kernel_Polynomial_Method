import numpy as np
import scipy


def mass_disorder(Wm, total_num_sites, seed):
    rng_generator = np.random.default_rng(seed=seed)
    if (total_num_sites / 2) == int(total_num_sites / 2):
        sbl_mass_disorder = rng_generator.uniform(low=-Wm/2, high=Wm/2, size=int(total_num_sites / 2))
    else:
        raise ValueError
    return scipy.sparse.diags(np.concatenate((sbl_mass_disorder, -1*sbl_mass_disorder)), format='csr', dtype=np.float64)


def bond_disorder(Wb, tbham, seed):
    if (tbham.nnz / 2) == int(tbham.nnz / 2):
        num_unique_bonds = int(tbham.nnz / 2)
    else:
        raise ValueError
    bond_rows, bond_cols = tbham.nonzero()
    redundancy_mask = bond_cols > bond_rows
    bond_rows = bond_rows[redundancy_mask]
    bond_cols = bond_cols[redundancy_mask]

    rng_generator = np.random.default_rng(seed=seed)
    disorders = rng_generator.uniform(low=-Wb/2, high=Wb/2, size=num_unique_bonds)
    disorder_mat = scipy.sparse.csr_array((np.tile(disorders, 2),
                                           (np.concatenate((bond_rows, bond_cols)),
                                            np.concatenate((bond_cols, bond_rows)))),
                                          shape=(tbham.shape[0], tbham.shape[1]))
    return disorder_mat

