from Lattice.General_Hamiltonian import *
from Lattice.General_Hamiltonian_Strain import sublattice_label_q3


def number_plaquets_q3(p, num_levels):
    plaquets_per_level = np.array([])
    number_plaquets = 0

    plaquets_per_threeside = (p-4)
    plaquets_per_fourside = (p-5)

    if num_levels >= 1:
        number_plaquets = number_plaquets + 1
        plaquets_per_level = np.append(plaquets_per_level, 1)

    if num_levels >= 2:
        number_plaquets = number_plaquets + p
        plaquets_per_level = np.append(plaquets_per_level, p)

    if num_levels >= 3:
        num_prev_threesides = p
        num_prev_foursides = 0
        for n in range(3, num_levels + 1):
            addterm = (num_prev_threesides * plaquets_per_threeside) + (num_prev_foursides * plaquets_per_fourside)
            number_plaquets = number_plaquets + addterm
            plaquets_per_level = np.append(plaquets_per_level, addterm)
            num_prev_threesides = (p-5)*num_prev_threesides + (p-6)*num_prev_foursides
            num_prev_foursides = plaquets_per_level[n-2]

    return plaquets_per_level, number_plaquets


def general_hyperbolic_q3_real_magnetic_field_hamiltonian(pval, num_levels, beta):
    hamiltonian_values = np.array([])
    hamiltonian_rows = np.array([])
    hamiltonian_cols = np.array([])
    first_site_each_gen = first_site_on_gen_hyperbolic_q3(pval, num_levels)
    t = 1
    prev_gen_ps_phases = None
    for nl in range(num_levels - 1):
        if nl == 0:
            sites_on_curr_gen = np.arange(pval)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
            prev_gen_ps_phases = np.repeat(beta / pval, pval)
            hamiltonian_values = np.append(hamiltonian_values, t * np.exp(-1j * prev_gen_ps_phases))
            hamiltonian_values = np.append(hamiltonian_values, t * np.exp(1j * prev_gen_ps_phases))

            sites_next_gen_conn_below = get_sites_conn_to_below_gen(pval, nl + 2, first_site_each_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_next_gen_conn_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_next_gen_conn_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
            hamiltonian_values = np.append(hamiltonian_values, np.repeat(t, int(2 * len(sites_on_curr_gen))))

        else:
            sites_on_curr_gen = np.arange(first_site_each_gen[nl], first_site_each_gen[nl + 1]).astype(np.int64)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
            if nl == 1:
                prev_gen_intervals = np.repeat(1, pval)
            else:
                prev_gen_conn_above = get_sites_conn_to_above_gen(pval, nl, first_site_each_gen)
                prev_gen_intervals = prev_gen_conn_above[1:] - prev_gen_conn_above[:-1]
                prev_gen_intervals = np.append(prev_gen_intervals, np.array([2]))
            prev_gen_intervals = np.append(np.array([0]), prev_gen_intervals)
            prev_gen_ps_phases_increment_sum = np.add.reduceat(np.roll(prev_gen_ps_phases, -1),
                                                               np.cumsum(prev_gen_intervals[:-1]).astype(np.int64))
            curr_gen_increments = np.tile(get_motif_of_sector_increment_q3(pval, nl + 1), pval)
            prev_gen_ps_phases = np.repeat((beta + prev_gen_ps_phases_increment_sum) / curr_gen_increments,
                                           curr_gen_increments.astype(np.int64))
            hamiltonian_values = np.append(hamiltonian_values, t * np.exp(-1j * prev_gen_ps_phases))
            hamiltonian_values = np.append(hamiltonian_values, t * np.exp(1j * prev_gen_ps_phases))

            sites_next_gen_conn_below = get_sites_conn_to_below_gen(pval, nl + 2, first_site_each_gen)
            sites_curr_gen_conn_above = get_sites_conn_to_above_gen(pval, nl + 1, first_site_each_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_curr_gen_conn_above)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_next_gen_conn_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_next_gen_conn_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_curr_gen_conn_above)
            hamiltonian_values = np.append(hamiltonian_values, np.repeat(t, int(2 * len(sites_curr_gen_conn_above))))

    sites_on_curr_gen = np.arange(first_site_each_gen[num_levels - 1],
                                  first_site_each_gen[num_levels]).astype(np.int64)
    hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
    hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
    hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
    hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
    prev_gen_conn_above = get_sites_conn_to_above_gen(pval, num_levels - 1, first_site_each_gen)
    prev_gen_intervals = prev_gen_conn_above[1:] - prev_gen_conn_above[:-1]
    prev_gen_intervals = np.append(prev_gen_intervals, np.array([2]))
    prev_gen_intervals = np.append(np.array([0]), prev_gen_intervals)
    prev_gen_ps_phases_increment_sum = np.add.reduceat(np.roll(prev_gen_ps_phases, -1),
                                                       np.cumsum(prev_gen_intervals[:-1]).astype(np.int64))
    curr_gen_increments = np.tile(get_motif_of_sector_increment_q3(pval, num_levels), pval)
    prev_gen_ps_phases = np.repeat((beta + prev_gen_ps_phases_increment_sum) / curr_gen_increments,
                                   curr_gen_increments.astype(np.int64))
    hamiltonian_values = np.append(hamiltonian_values, t * np.exp(-1j * prev_gen_ps_phases))
    hamiltonian_values = np.append(hamiltonian_values, t * np.exp(1j * prev_gen_ps_phases))

    total_num_sites = number_points_q3_general_from_repeating_pattern(pval, num_levels)[1]
    hamiltonian = csr_array((hamiltonian_values,
                             (hamiltonian_rows.astype(np.int64), hamiltonian_cols.astype(np.int64))),
                            shape=(total_num_sites, total_num_sites))

    return hamiltonian


def general_q3_hamiltonian_with_real_B_field_NH(p, num_levels, beta, alpha_NH):
    ham = general_hyperbolic_q3_real_magnetic_field_hamiltonian(p, num_levels, beta)
    asites, bsites = sublattice_label_q3(p, num_levels)
    sublat_basis = np.concatenate((asites, bsites))

    ham = ham.tocsr()
    ham = ham[:, sublat_basis][sublat_basis, :]
    ham = ham.multiply(np.concatenate((np.repeat(1 + alpha_NH, int(ham.shape[0] / 2)),
                                       np.repeat(1 - alpha_NH, int(ham.shape[0] / 2)))).reshape(-1, 1))

    return ham
