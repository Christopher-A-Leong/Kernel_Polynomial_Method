from Lattice.General_Hamiltonian import *


def sublattice_label_q3(pval, nval):
    points_per_level, total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)
    first_site_on_gen = np.cumsum(np.append(np.array([0]), points_per_level))
    asites = np.array([])
    bsites = np.array([])

    if nval >= 1:
        asites = np.append(asites, np.arange(first_site_on_gen[0], first_site_on_gen[1], 2))
        bsites = np.append(bsites, np.arange(first_site_on_gen[0] + 1, first_site_on_gen[1], 2))
    if nval > 1:
        for nl in range(2, nval + 1):
            asites = np.append(asites, np.arange(first_site_on_gen[nl-1] + 1, first_site_on_gen[nl], 2))
            bsites = np.append(bsites, np.arange(first_site_on_gen[nl-1], first_site_on_gen[nl], 2))

    return asites.astype(int), bsites.astype(int)


def get_interval_rvals(rstart, rend, num_sites):
    rvals = np.zeros(num_sites)
    rvals[0] = rstart
    rvals[-1] = rend
    if num_sites % 2 == 1:
        rvals[:int((len(rvals) - 1)/2)] = np.arange(rstart, rstart + int((len(rvals) - 1)/2))
        rvals[int((len(rvals) - 1)/2 + 1):] = np.flip(np.arange(rend, rend + int((len(rvals) - 1)/2)))
        rvals[int((num_sites - 1) / 2)] = np.max(rvals) + 1
    else:
        rvals[:int(len(rvals)/2)] = np.arange(rstart, rstart + int(len(rvals)/2))
        rvals[int(len(rvals) / 2):] = np.flip(np.arange(rend, rend + int(len(rvals) / 2)))
    return rvals


def get_rvals_on_gen(pval, rvals_curr_gen_conn_below, interval_num_sites):
    if len(rvals_curr_gen_conn_below) / pval != int(len(rvals_curr_gen_conn_below) / pval):
        raise ValueError
    sector_rval_anchors = rvals_curr_gen_conn_below[:int(len(rvals_curr_gen_conn_below) / pval + 1)]
    rvals = np.concatenate(
        [get_interval_rvals(sector_rval_anchors[i], sector_rval_anchors[i + 1], interval_num_sites[i])[:-1]
         for i in range(len(sector_rval_anchors) - 1)])
    return np.tile(rvals, pval)


def general_hyperbolic_q3_pseudo_magnetic_field_hamiltonian(pval, nval, qaxial):

    hamiltonian_rows = np.array([])
    hamiltonian_cols = np.array([])
    hamiltonian_values = np.array([])
    first_site_on_each_gen = first_site_on_gen_hyperbolic_q3(pval, nval)
    asites, bsites = sublattice_label_q3(pval, nval)
    sublattice_sign_hash_table = {i: sbl_sgn
                                  for i, sbl_sgn in zip(np.concatenate((asites, bsites)),
                                                        np.concatenate((np.repeat(1, len(asites)),
                                                                        np.repeat(-1, len(bsites)))))}
    t = 1
    next_gen_num_sites_interval = None
    next_gen_conn_below_rvals = None
    for nl in range(nval-1):
        if nl == 0:
            sites_on_curr_gen = np.arange(pval)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
            hamiltonian_values = np.append(hamiltonian_values, np.repeat(t, int(2*pval)))

            next_gen_conn_below = get_sites_conn_to_below_gen(pval, nl + 2, first_site_on_each_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, next_gen_conn_below)
            hamiltonian_cols = np.append(hamiltonian_cols, next_gen_conn_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
            curr_gen_rvals = np.repeat(0, pval)
            next_gen_rvals = np.repeat(1, pval)
            sign_factor = np.array([sublattice_sign_hash_table[i] for i in sites_on_curr_gen])
            exp_factor = (sign_factor * curr_gen_rvals ** 2) + (-1 * sign_factor * next_gen_rvals ** 2)
            hamiltonian_values = np.append(hamiltonian_values,
                                           np.tile(t * np.exp(qaxial * exp_factor), 2))
            next_gen_num_sites_interval = (next_gen_conn_below[1:] - next_gen_conn_below[:-1]) + 1
            next_gen_conn_below_rvals = next_gen_rvals.copy()
        else:
            sites_on_curr_gen = np.arange(first_site_on_each_gen[nl], first_site_on_each_gen[nl + 1])
            curr_gen_rvals = get_rvals_on_gen(pval, next_gen_conn_below_rvals,
                                              next_gen_num_sites_interval.astype(np.int64))
            hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
            sign_factor = np.array([sublattice_sign_hash_table[i] for i in sites_on_curr_gen])
            exp_factor = ((sign_factor * curr_gen_rvals ** 2)
                          + (np.roll(sign_factor, -1) * np.roll(curr_gen_rvals, -1) ** 2))
            hamiltonian_values = np.append(hamiltonian_values,
                                           np.tile(t * np.exp(qaxial * exp_factor), 2))

            next_gen_conn_below = get_sites_conn_to_below_gen(pval, nl + 2, first_site_on_each_gen)
            curr_gen_conn_above = get_sites_conn_to_above_gen(pval, nl + 1, first_site_on_each_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, curr_gen_conn_above)
            hamiltonian_rows = np.append(hamiltonian_rows, next_gen_conn_below)
            hamiltonian_cols = np.append(hamiltonian_cols, next_gen_conn_below)
            hamiltonian_cols = np.append(hamiltonian_cols, curr_gen_conn_above)
            curr_gen_rvals = curr_gen_rvals[curr_gen_conn_above - first_site_on_each_gen[nl]]
            sign_factor = np.array([sublattice_sign_hash_table[i] for i in curr_gen_conn_above])
            exp_factor = (sign_factor * curr_gen_rvals ** 2) + (-1 * sign_factor * (curr_gen_rvals + 1) ** 2)
            hamiltonian_values = np.append(hamiltonian_values, np.tile(t * np.exp(qaxial * exp_factor), 2))

            next_gen_num_sites_interval = (next_gen_conn_below[1:] - next_gen_conn_below[:-1]) + 1
            next_gen_conn_below_rvals = curr_gen_rvals + 1

    sites_on_curr_gen = np.arange(first_site_on_each_gen[nval - 1], first_site_on_each_gen[nval])
    curr_gen_rvals = get_rvals_on_gen(pval, next_gen_conn_below_rvals,
                                      next_gen_num_sites_interval.astype(np.int64))
    hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
    hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
    hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
    hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
    sign_factor = np.array([sublattice_sign_hash_table[i] for i in sites_on_curr_gen])
    exp_factor = ((sign_factor * curr_gen_rvals ** 2)
                  + (np.roll(sign_factor, -1) * np.roll(curr_gen_rvals, -1) ** 2))
    hamiltonian_values = np.append(hamiltonian_values,
                                   np.tile(t * np.exp(qaxial * exp_factor), 2))

    total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)[1]
    hamiltonian = csr_array((hamiltonian_values,
                             (hamiltonian_rows.astype(np.int64), hamiltonian_cols.astype(np.int64))),
                            shape=(total_num_sites, total_num_sites))

    return hamiltonian


def general_q3_hamiltonian_with_strain_nonHermitian(pval, nval, qax, alpha_NH):
    ham = general_hyperbolic_q3_pseudo_magnetic_field_hamiltonian(pval, nval, qax)
    asites, bsites = sublattice_label_q3(pval, nval)
    sublat_basis = np.concatenate((asites, bsites))

    ham = ham.tocsr()
    ham = ham[:, sublat_basis][sublat_basis, :]
    ham = ham.multiply(np.concatenate((np.repeat(1 + alpha_NH, int(ham.shape[0] / 2)),
                                       np.repeat(1 - alpha_NH, int(ham.shape[0] / 2)))).reshape(-1, 1))

    return ham

