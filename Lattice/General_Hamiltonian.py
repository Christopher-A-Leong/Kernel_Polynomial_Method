import numpy as np
from scipy.sparse import csr_array


def number_points_q3_general_from_repeating_pattern(p, num_levels):
    sites_per_level = np.array([])
    number_prev_3sides = None
    number_prev_4sides = None
    for n in range(num_levels):
        if n == 0:
            sites_per_level = np.append(sites_per_level, p)
        elif n == 1:
            sites_per_level = np.append(sites_per_level, p*(p-3))
            number_prev_3sides = p
            number_prev_4sides = 0
        else:
            number_3sides = (p-5)*number_prev_3sides + (p-6)*number_prev_4sides
            number_4sides = number_prev_3sides + number_prev_4sides
            sites_per_level = np.append(sites_per_level, number_3sides*(p-3) + number_4sides*(p-4))
            number_prev_3sides = number_3sides
            number_prev_4sides = number_4sides
    sites_per_level = np.array([int(i) for i in sites_per_level])
    return sites_per_level, np.sum(sites_per_level)


def first_site_on_gen_hyperbolic_q3(pval, num_levels):
    points_per_level = number_points_q3_general_from_repeating_pattern(pval, num_levels)[0]
    points_per_level = np.append(np.array([0.0]), points_per_level)
    return np.cumsum(points_per_level).astype(np.int64)


def get_motif_of_sector_symbolic_q3(p, specific_level):
    ontop_3side_motif = np.concatenate((np.repeat('3side', (p-5)), np.array(['4side'])))
    ontop_4side_motif = np.concatenate((np.repeat('3side', (p-6)), np.array(['4side'])))
    ontop_lookup_table = {'3side': ontop_3side_motif.tolist(),
                          '4side': ontop_4side_motif.tolist()}
    new_motif = None
    prev_motif = None
    for n in range(1, specific_level):
        if n == 1:
            new_motif = ['3side']
            prev_motif = ['3side']
        else:
            new_motif = [ontop_lookup_table[v] for v in prev_motif]
            new_motif = np.concatenate(new_motif).tolist()
            prev_motif = new_motif.copy()
    return new_motif


def get_motif_of_sector_increment_q3(p, specific_level):
    symbolic_motif = np.array(get_motif_of_sector_symbolic_q3(p, specific_level))
    increment_motif = np.zeros(len(symbolic_motif))
    increment_motif[symbolic_motif == '3side'] = p - 3
    increment_motif[symbolic_motif == '4side'] = p - 4
    return increment_motif


def get_sites_conn_to_below_gen(pval, current_gen, first_sites_on_gen):
    motif_increment = get_motif_of_sector_increment_q3(pval, current_gen)
    current_gen = current_gen - 1
    calculation_setup = np.append(np.array([first_sites_on_gen[current_gen]]),
                                  np.tile(motif_increment, pval))
    return np.cumsum(calculation_setup)[:-1]


def get_sites_conn_to_above_gen(pval, current_gen, first_sites_on_gen):
    if current_gen == 1:
        return np.arange(pval).astype(np.int64)
    else:
        sites_conn_below_gen = get_sites_conn_to_below_gen(pval, current_gen, first_sites_on_gen).astype(np.int64)
        current_gen = current_gen - 1
        all_sites_on_gen = np.arange(first_sites_on_gen[current_gen], first_sites_on_gen[current_gen + 1]).astype(np.int64)
        mask = np.repeat(True, len(all_sites_on_gen))
        mask[sites_conn_below_gen - first_sites_on_gen[current_gen]] = False
        return all_sites_on_gen[mask]


def general_hyperbolic_q3_hamiltonian(pval, num_levels):
    hamiltonian_rows = np.array([])
    hamiltonian_cols = np.array([])
    first_site_on_each_gen = first_site_on_gen_hyperbolic_q3(pval, num_levels)
    for nl in range(num_levels - 1):
        if nl == 0:
            sites_on_curr_gen = np.arange(first_site_on_each_gen[nl], first_site_on_each_gen[nl + 1]).astype(np.int64)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
            sites_curr_conn_to_above = get_sites_conn_to_above_gen(pval, nl + 1, first_site_on_each_gen)
            sites_next_conn_to_below = get_sites_conn_to_below_gen(pval, nl + 2, first_site_on_each_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_curr_conn_to_above)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_next_conn_to_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_next_conn_to_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_curr_conn_to_above)

        else:
            sites_on_curr_gen = np.arange(first_site_on_each_gen[nl], first_site_on_each_gen[nl + 1]).astype(np.int64)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
            hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)
            sites_curr_conn_to_above = get_sites_conn_to_above_gen(pval, nl + 1, first_site_on_each_gen)
            sites_next_conn_to_below = get_sites_conn_to_below_gen(pval, nl + 2, first_site_on_each_gen)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_curr_conn_to_above)
            hamiltonian_rows = np.append(hamiltonian_rows, sites_next_conn_to_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_next_conn_to_below)
            hamiltonian_cols = np.append(hamiltonian_cols, sites_curr_conn_to_above)

    sites_on_curr_gen = np.arange(first_site_on_each_gen[num_levels - 1],
                                  first_site_on_each_gen[num_levels]).astype(np.int64)
    hamiltonian_rows = np.append(hamiltonian_rows, sites_on_curr_gen)
    hamiltonian_rows = np.append(hamiltonian_rows, np.roll(sites_on_curr_gen, -1))
    hamiltonian_cols = np.append(hamiltonian_cols, np.roll(sites_on_curr_gen, -1))
    hamiltonian_cols = np.append(hamiltonian_cols, sites_on_curr_gen)

    total_num_sites = number_points_q3_general_from_repeating_pattern(pval, num_levels)[1]
    hamiltonian = csr_array((np.ones(len(hamiltonian_rows)),
                            (hamiltonian_rows.astype(np.int64), hamiltonian_cols.astype(np.int64))),
                            shape=(total_num_sites, total_num_sites))
    return hamiltonian

