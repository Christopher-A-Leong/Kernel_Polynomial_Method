import numpy as np
from Lattice.General_Hamiltonian import number_points_q3_general_from_repeating_pattern
from Lattice.General_Hamiltonian_Strain import sublattice_label_q3
from Lattice.General_Hamiltonian_q4 import points_that_connect_to_below_q4, points_that_connect_to_above_q4
from Lattice.General_Hamiltonian_q4 import sites_on_level_q4, sublattice_label_q4


def sites_on_each_gen_hyperbolic_q3(pval, num_levels):
    sites_per_level, total_num_sites = number_points_q3_general_from_repeating_pattern(pval, num_levels)
    sites_per_level = np.append(np.array([0]), sites_per_level)
    sites_on_each_gen = {}
    for n in range(1, num_levels + 1):
        sites_on_each_gen['n={}'.format(n - 1)] = np.arange(np.sum(sites_per_level[:n]),
                                                            np.sum(sites_per_level[:n + 1]))
    return sites_on_each_gen


def search_for_unique_plaquettes_each_gen_hyperbolic_q3(pval, num_levels):
    gen_sites_dict = sites_on_each_gen_hyperbolic_q3(pval, num_levels)
    plaquette_indices = np.zeros((1, pval))
    offset_low = 0
    offset_high = 0
    for n in range(num_levels - 1):
        lower_gen_sites = gen_sites_dict['n={}'.format(n)][int(offset_low):int(offset_low + 2)]
        higher_gen_sites = gen_sites_dict['n={}'.format(n + 1)][int(offset_high):int(offset_high + (pval - 2))]
        plaquette_indices = np.vstack((plaquette_indices,
                                       np.concatenate((lower_gen_sites, higher_gen_sites))))

        offset_low = (n + 1) * (len(gen_sites_dict['n={}'.format(n + 1)]) / pval) + 1
        if n != num_levels - 2:
            offset_high = (n + 1) * (len(gen_sites_dict['n={}'.format(n + 2)]) / pval)

    plaquette_indices = plaquette_indices.astype(np.int64)
    return np.concat((np.arange(0, pval, dtype=np.int64), plaquette_indices[1:, :].reshape(-1)))


def search_for_unique_plaquettes_each_gen_sublattice_basis_hyperbolic_q3(pval, num_levels):
    plaquettes_each_gen_sites = search_for_unique_plaquettes_each_gen_hyperbolic_q3(pval, num_levels)
    asites, bsites = sublattice_label_q3(pval, num_levels)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    site_label_map = np.argsort(sublattice_basis)

    plaquettes_each_gen_sites_sublattice_basis = np.array([])
    for i in plaquettes_each_gen_sites:
        plaquettes_each_gen_sites_sublattice_basis = np.append(plaquettes_each_gen_sites_sublattice_basis,
                                                               site_label_map[i])

    return plaquettes_each_gen_sites_sublattice_basis


def search_for_unique_plaquettes_each_gen_hyperbolic_q4(pval, num_levels):
    plaquettes_each_gen = np.zeros((num_levels, pval), dtype=np.int64)
    plaquettes_each_gen[0, :] = np.arange(pval, dtype=np.int64)

    offset_counter = 0
    for n in range(1, num_levels):
        if n == 1:
            below_gen_conn_above = points_that_connect_to_above_q4(pval, n)
            curr_gen_conn_below = points_that_connect_to_below_q4(pval, n + 1)
            plaquettes_each_gen[n, :] = np.concat((below_gen_conn_above[:2],
                                                   np.arange(curr_gen_conn_below[0], curr_gen_conn_below[1] + 1)
                                                   ))
            offset_counter += 1
        else:
            if (len(sites_on_level_q4(pval, n + 1)) % pval != 0) or (len(sites_on_level_q4(pval, n)) % pval != 0):
                raise ValueError
            offset_interval_below_gen = int(len(sites_on_level_q4(pval, n)) / pval)
            offset_interval_curr_gen = int(len(sites_on_level_q4(pval, n + 1)) / pval)
            below_gen_conn_above = points_that_connect_to_above_q4(pval, n)
            curr_gen_conn_below = points_that_connect_to_below_q4(pval, n + 1)
            plaquettes_each_gen[n, :] = np.concat((below_gen_conn_above[:2] + offset_counter * offset_interval_below_gen,
                                                   np.arange(curr_gen_conn_below[0], curr_gen_conn_below[1] + 1)
                                                   + offset_counter * offset_interval_curr_gen
                                                   ))
            offset_counter += 1

    return plaquettes_each_gen.reshape(-1)


def search_for_unique_plaquettes_each_gen_sublattice_basis_hyperbolic_q4(pval, num_levels):
    plaquettes_each_gen_sites = search_for_unique_plaquettes_each_gen_hyperbolic_q4(pval, num_levels)
    asites, bsites = sublattice_label_q4(pval, num_levels)
    sublattice_basis = np.concat((asites, bsites)).astype(np.int64)
    site_label_map = np.argsort(sublattice_basis)

    plaquettes_each_gen_sites_sublattice_basis = np.array([])
    for i in plaquettes_each_gen_sites:
        plaquettes_each_gen_sites_sublattice_basis = np.append(plaquettes_each_gen_sites_sublattice_basis,
                                                               site_label_map[i])

    return plaquettes_each_gen_sites_sublattice_basis
