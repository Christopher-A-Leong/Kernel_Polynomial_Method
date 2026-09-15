import numpy as np
import scipy


def identify_nearest_neighbors(nntbham):
    nz_rowind, nz_colind = nntbham.nonzero()
    return nz_rowind, nz_colind


def identify_next_nearest_neighbors(nntbham):
    adjacency_sq = nntbham @ nntbham
    adjacency_sq = adjacency_sq - scipy.sparse.diags_array(adjacency_sq.diagonal().astype(np.float64),
                                                           offsets=0, format='csr')
    adjacency_sq.eliminate_zeros()
    nz_rowind, nz_colind = adjacency_sq.nonzero()
    return nz_rowind, nz_colind


def convert_neighbors_list_to_hash_table(nz_rowind, nz_colind):
    nz_colind_sorted = nz_colind[np.argsort(nz_rowind)]
    nz_rowind_sorted = np.sort(nz_rowind)
    nz_rowind_unique, row_ind_split_lines = np.unique(nz_rowind_sorted, return_index=True)
    nz_colind_groupby_rowind = np.split(nz_colind_sorted, row_ind_split_lines[1:])
    hash_table = {r: c for (r, c) in zip(nz_rowind_unique, nz_colind_groupby_rowind)}
    return hash_table


def get_neighbors_of_site(site_index, nz_hash_table):
    return nz_hash_table[site_index]
