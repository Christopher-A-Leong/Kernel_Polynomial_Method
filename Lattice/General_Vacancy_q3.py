import numpy as np
from Lattice.General_Hamiltonian_Strain import sublattice_label_q3
from Lattice.General_Hamiltonian import general_hyperbolic_q3_hamiltonian
from Lattice.General_Hamiltonian import number_points_q3_general_from_repeating_pattern
from Operations.Neighbors import identify_nearest_neighbors
from Operations.Neighbors import identify_next_nearest_neighbors
from Operations.Neighbors import get_neighbors_of_site
from Operations.Neighbors import convert_neighbors_list_to_hash_table
from Lattice.Honeycomb_Sparse import honeycomb_lattice_sparse_PBC
from Lattice.Honeycomb_Sparse import honeycomb_points
from Lattice.Honeycomb_Sparse import honeycomb_site_assignment


def check_for_isolated_sites(vacancy_ham):
    num_isolated_points = vacancy_ham.shape[0] - len(np.unique(vacancy_ham.indices))
    if num_isolated_points == 0:
        return False
    else:
        return True


def check_equal_sublattice_vd(vacant_sites, asites, bsites):
    num_vacant_asites = np.sum(np.isin(vacant_sites, asites))
    num_vacant_bsites = np.sum(np.isin(vacant_sites, bsites))
    return num_vacant_asites == num_vacant_bsites


def hyperbolic_q3_equal_sublattice_vacancy_density_legacy_rng(pval, nval, vacancy_density,
                                                              seed=None, isocheck=False):
    if (pval % 2) != 0:
        raise ValueError
    sparse_ham = general_hyperbolic_q3_hamiltonian(pval, nval)

    total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)[1]
    num_sublat_sites = int(total_num_sites / 2)
    num_vacancies_per_sublat = int(round(vacancy_density * num_sublat_sites, 0))

    asites, bsites = sublattice_label_q3(pval, nval)
    if seed is None:
        asite_vacancies = np.random.choice(asites, size=num_vacancies_per_sublat, replace=False)
        bsite_vacancies = np.random.choice(bsites, size=num_vacancies_per_sublat, replace=False)
    else:
        rng = np.random.RandomState(seed=seed)
        asite_vacancies = rng.choice(asites, size=num_vacancies_per_sublat, replace=False)
        bsite_vacancies = rng.choice(bsites, size=num_vacancies_per_sublat, replace=False)

    all_site_vacancies = np.concatenate((asite_vacancies, bsite_vacancies))
    non_vacancy_inds = np.setdiff1d(np.arange(total_num_sites).astype(np.int64), all_site_vacancies)

    sparse_ham = sparse_ham.tocsr()
    sparse_ham = sparse_ham[non_vacancy_inds, :][:, non_vacancy_inds]

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham)
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))

    return sparse_ham


def hyperbolic_q3_equal_sublattice_vacancy_density_bulk_only_legacy_rng(pval, nval, bulk_vacancy_density,
                                                                        seed=None, isocheck=False):
    if (pval % 2) != 0:
        raise ValueError
    sparse_ham = general_hyperbolic_q3_hamiltonian(pval, nval)

    points_per_level, total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)
    num_sublat_bulk_sites = int(np.sum(points_per_level[:-1]) / 2)
    num_bulk_vacancies_per_sublat = int(round(bulk_vacancy_density * num_sublat_bulk_sites, 0))

    asites, bsites = sublattice_label_q3(pval, nval)
    bulk_asites = asites[asites < np.sum(points_per_level[:-1])]
    bulk_bsites = bsites[bsites < np.sum(points_per_level[:-1])]

    if seed is None:
        bulk_asite_vacancies = np.random.choice(bulk_asites, size=num_bulk_vacancies_per_sublat, replace=False)
        bulk_bsite_vacancies = np.random.choice(bulk_bsites, size=num_bulk_vacancies_per_sublat, replace=False)
    else:
        rng = np.random.RandomState(seed=seed)
        bulk_asite_vacancies = rng.choice(bulk_asites, size=num_bulk_vacancies_per_sublat, replace=False)
        bulk_bsite_vacancies = rng.choice(bulk_bsites, size=num_bulk_vacancies_per_sublat, replace=False)

    all_site_vacancies = np.concatenate((bulk_asite_vacancies, bulk_bsite_vacancies))
    non_vacancy_inds = np.setdiff1d(np.arange(total_num_sites).astype(np.int64), all_site_vacancies)

    sparse_ham = sparse_ham.tocsr()
    sparse_ham = sparse_ham[non_vacancy_inds, :][:, non_vacancy_inds]

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham)
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))

    return sparse_ham


def hyperbolic_q3_equal_sublattice_vacancy_density_distance_restriction_legacy_rng(pval, nval, vacancy_density,
                                                                                   seed, isocheck=False):
    if (pval % 2) != 0:
        raise ValueError
    sparse_ham = general_hyperbolic_q3_hamiltonian(pval, nval)

    total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)[1]
    num_sublat_sites = int(total_num_sites / 2)
    num_vacancies_per_sublat = int(round(vacancy_density * num_sublat_sites, 0))

    asites, bsites = sublattice_label_q3(pval, nval)

    all_site_vacancies = np.repeat(-1.0, int(2*num_vacancies_per_sublat))
    rng = np.random.RandomState(seed=seed)
    if len(all_site_vacancies) <= 100000:
        random_seeds = rng.randint(low=1, high=10000000, size=100000)
    elif len(all_site_vacancies) <= 1000000:
        random_seeds = rng.randint(low=1, high=100000000, size=1000000)
    else:
        random_seeds = rng.randint(low=1, high=1000000000, size=10000000)
    while_loop_counter = 0
    nn_rows, nn_cols = identify_nearest_neighbors(sparse_ham)
    nn_hash_table = convert_neighbors_list_to_hash_table(nn_rows, nn_cols)
    nnn_rows, nnn_cols = identify_next_nearest_neighbors(sparse_ham)
    nnn_hash_table = convert_neighbors_list_to_hash_table(nnn_rows, nnn_cols)
    exclusion_mask_asites = np.repeat(True, len(asites))
    exclusion_mask_bsites = np.repeat(True, len(bsites))
    new_excluded_asites = np.zeros(10, dtype=np.int64)
    new_excluded_bsites = np.zeros(10, dtype=np.int64)

    hash_table_asites = {v: i for (v, i) in zip(asites, np.arange(len(asites), dtype=np.int64))}
    hash_table_bsites = {v: i for (v, i) in zip(bsites, np.arange(len(bsites), dtype=np.int64))}

    while while_loop_counter < (2 * num_vacancies_per_sublat):
        curr_rng = np.random.RandomState(seed=random_seeds[while_loop_counter])
        trial_vac_asite = curr_rng.choice(asites[exclusion_mask_asites], size=1)

        new_excluded_asites[:] = -1
        new_excluded_bsites[:] = -1
        nn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nn_hash_table)
        nnn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nnn_hash_table)
        new_excluded_asites[0] = trial_vac_asite[0]
        new_excluded_bsites[:len(nn_neighbors)] = nn_neighbors
        new_excluded_asites[1:len(nnn_neighbors) + 1] = nnn_neighbors

        # exclusion_mask_bsites[np.isin(bsites, new_excluded_bsites)] = False
        exclusion_mask_bsites[[hash_table_bsites[val] for val in new_excluded_bsites[new_excluded_bsites != -1]]] = False

        while_loop_counter += 1

        curr_rng = np.random.RandomState(seed=random_seeds[while_loop_counter])
        trial_vac_bsite = curr_rng.choice(bsites[exclusion_mask_bsites], size=1)

        more_nn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nn_hash_table)
        more_nnn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nnn_hash_table)
        new_excluded_bsites[len(nn_neighbors)] = trial_vac_bsite[0]
        new_excluded_asites[len(nnn_neighbors) + 1:len(nnn_neighbors) + 1 + len(more_nn_neighbors)] = more_nn_neighbors
        new_excluded_bsites[len(nn_neighbors) + 1:len(nn_neighbors) + 1 + len(more_nnn_neighbors)] = more_nnn_neighbors

        # exclusion_mask_asites[np.isin(asites, new_excluded_asites)] = False
        # exclusion_mask_bsites[np.isin(bsites, new_excluded_bsites)] = False
        exclusion_mask_asites[[hash_table_asites[val] for val in new_excluded_asites[new_excluded_asites != -1]]] = False
        exclusion_mask_bsites[[hash_table_bsites[val] for val in new_excluded_bsites[new_excluded_bsites != -1]]] = False

        while_loop_counter += 1

        # all_site_vacancies = np.append(all_site_vacancies, trial_vac_asite)
        # all_site_vacancies = np.append(all_site_vacancies, trial_vac_bsite)
        all_site_vacancies[while_loop_counter-2] = trial_vac_asite[0]
        all_site_vacancies[while_loop_counter-1] = trial_vac_bsite[0]

    non_vacancy_inds = np.setdiff1d(np.arange(total_num_sites).astype(np.int64), all_site_vacancies)
    sparse_ham = sparse_ham.tocsr()
    sparse_ham = sparse_ham[non_vacancy_inds, :][:, non_vacancy_inds]

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham)
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))

    return sparse_ham


def hyperbolic_q3_equal_sublattice_vacancy_density_bulk_only_distance_restriction_legacy_rng(pval, nval, vacancy_density,
                                                                                             seed, isocheck=False,
                                                                                             equal_vd_check=False):
    if (pval % 2) != 0:
        raise ValueError
    sparse_ham = general_hyperbolic_q3_hamiltonian(pval, nval)

    points_per_level, total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)
    bulk_cutoff = int(np.sum(points_per_level[:-1]))
    num_bulk_sublat_sites = int(bulk_cutoff / 2)
    num_bulk_vacancies_per_sublat = int(round(vacancy_density * num_bulk_sublat_sites, 0))

    asites, bsites = sublattice_label_q3(pval, nval)
    bulk_asites = asites[asites < bulk_cutoff]
    bulk_bsites = bsites[bsites < bulk_cutoff]

    all_site_vacancies = np.repeat(-1.0, int(2*num_bulk_vacancies_per_sublat))
    rng = np.random.RandomState(seed=seed)
    if len(all_site_vacancies) <= 100000:
        random_seeds = rng.randint(low=1, high=10000000, size=100000)
    elif len(all_site_vacancies) <= 1000000:
        random_seeds = rng.randint(low=1, high=100000000, size=1000000)
    else:
        random_seeds = rng.randint(low=1, high=1000000000, size=10000000)
    while_loop_counter = 0
    nn_rows, nn_cols = identify_nearest_neighbors(sparse_ham)
    nn_hash_table = convert_neighbors_list_to_hash_table(nn_rows, nn_cols)
    nnn_rows, nnn_cols = identify_next_nearest_neighbors(sparse_ham)
    nnn_hash_table = convert_neighbors_list_to_hash_table(nnn_rows, nnn_cols)
    exclusion_mask_asites = np.repeat(True, len(bulk_asites))
    exclusion_mask_bsites = np.repeat(True, len(bulk_bsites))
    new_excluded_asites = np.zeros(10, dtype=np.int64)
    new_excluded_bsites = np.zeros(10, dtype=np.int64)

    hash_table_asites = {v: i for (v, i) in zip(bulk_asites, np.arange(len(bulk_asites), dtype=np.int64))}
    hash_table_bsites = {v: i for (v, i) in zip(bulk_bsites, np.arange(len(bulk_bsites), dtype=np.int64))}

    while while_loop_counter < (2 * num_bulk_vacancies_per_sublat):
        curr_rng = np.random.RandomState(seed=random_seeds[while_loop_counter])
        trial_vac_asite = curr_rng.choice(bulk_asites[exclusion_mask_asites], size=1)

        new_excluded_asites[:] = -1
        new_excluded_bsites[:] = -1
        nn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nn_hash_table)
        nnn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nnn_hash_table)
        new_excluded_asites[0] = trial_vac_asite[0]
        new_excluded_bsites[:len(nn_neighbors)] = nn_neighbors
        new_excluded_asites[1:len(nnn_neighbors) + 1] = nnn_neighbors

        exclusion_mask_bsites[
            [hash_table_bsites[val]
             for val in new_excluded_bsites[np.logical_and(new_excluded_bsites != -1,
                                                           new_excluded_bsites < bulk_cutoff)]]
        ] = False

        while_loop_counter += 1

        curr_rng = np.random.RandomState(seed=random_seeds[while_loop_counter])
        trial_vac_bsite = curr_rng.choice(bulk_bsites[exclusion_mask_bsites], size=1)

        more_nn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nn_hash_table)
        more_nnn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nnn_hash_table)
        new_excluded_bsites[len(nn_neighbors)] = trial_vac_bsite[0]
        new_excluded_asites[len(nnn_neighbors) + 1:len(nnn_neighbors) + 1 + len(more_nn_neighbors)] = more_nn_neighbors
        new_excluded_bsites[len(nn_neighbors) + 1:len(nn_neighbors) + 1 + len(more_nnn_neighbors)] = more_nnn_neighbors

        exclusion_mask_asites[
            [hash_table_asites[val]
             for val in new_excluded_asites[np.logical_and(new_excluded_asites != -1,
                                                           new_excluded_asites < bulk_cutoff)]]
        ] = False
        exclusion_mask_bsites[
            [hash_table_bsites[val]
             for val in new_excluded_bsites[np.logical_and(new_excluded_bsites != -1,
                                                           new_excluded_bsites < bulk_cutoff)]]
        ] = False

        while_loop_counter += 1

        all_site_vacancies[while_loop_counter-2] = trial_vac_asite[0]
        all_site_vacancies[while_loop_counter-1] = trial_vac_bsite[0]

    if equal_vd_check:
        check_result = check_equal_sublattice_vd(all_site_vacancies, asites, bsites)
        print('Is equal vacancy density on each sublattice maintained: {}'.format(check_result))

    non_vacancy_inds = np.setdiff1d(np.arange(total_num_sites).astype(np.int64), all_site_vacancies)
    sparse_ham = sparse_ham.tocsr()
    sparse_ham = sparse_ham[non_vacancy_inds, :][:, non_vacancy_inds]

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham)
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))

    return sparse_ham


def hyperbolic_q3_number_nonvacant_sites_bulk_only_distance_restriction(pval, nval, vacancy_density):
    points_per_level, total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)
    num_bulk_sublat_points = np.sum(points_per_level[:-1]) / 2
    return total_num_sites - 2 * round(vacancy_density * num_bulk_sublat_points, 0)


def honeycomb_pbc_flake_number_nonvacant_sites_distance_restriction(nval, vacancy_density):
    total_num_sites = honeycomb_points(nval)[1]
    num_sublat_sites = int(total_num_sites / 2)
    num_vacancies_on_sublat = round(vacancy_density * num_sublat_sites, 0)
    return total_num_sites - 2 * num_vacancies_on_sublat


def honeycomb_pbc_flake_equal_sublattice_vacancy_density_distance_restriction_legacy_rng(nval, vacancy_density, seed,
                                                                                         isocheck=False,
                                                                                         equal_vd_check=False):
    sparse_ham = honeycomb_lattice_sparse_PBC(nval)

    total_num_sites = honeycomb_points(nval)[1]
    num_sublat_sites = int(total_num_sites / 2)
    num_vacancies_on_sublat = round(vacancy_density * num_sublat_sites, 0)

    asites, bsites = honeycomb_site_assignment(nval)

    all_site_vacancies = np.repeat(-1.0, int(2 * num_vacancies_on_sublat))
    rng = np.random.RandomState(seed=seed)
    if len(all_site_vacancies) <= 100000:
        random_seeds = rng.randint(low=1, high=10000000, size=100000)
    elif len(all_site_vacancies) <= 1000000:
        random_seeds = rng.randint(low=1, high=100000000, size=1000000)
    else:
        random_seeds = rng.randint(low=1, high=1000000000, size=10000000)
    while_loop_counter = 0
    nn_rows, nn_cols = identify_nearest_neighbors(sparse_ham)
    nn_hash_table = convert_neighbors_list_to_hash_table(nn_rows, nn_cols)
    nnn_rows, nnn_cols = identify_next_nearest_neighbors(sparse_ham)
    nnn_hash_table = convert_neighbors_list_to_hash_table(nnn_rows, nnn_cols)
    exclusion_mask_asites = np.repeat(True, len(asites))
    exclusion_mask_bsites = np.repeat(True, len(bsites))
    new_excluded_asites = np.zeros(10, dtype=np.int64)
    new_excluded_bsites = np.zeros(10, dtype=np.int64)

    hash_table_asites = {v: i for (v, i) in zip(asites, np.arange(len(asites), dtype=np.int64))}
    hash_table_bsites = {v: i for (v, i) in zip(bsites, np.arange(len(bsites), dtype=np.int64))}

    while while_loop_counter < (2 * num_vacancies_on_sublat):
        curr_rng = np.random.RandomState(seed=random_seeds[while_loop_counter])
        trial_vac_asite = curr_rng.choice(asites[exclusion_mask_asites], size=1)

        new_excluded_asites[:] = -1
        new_excluded_bsites[:] = -1
        nn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nn_hash_table)
        nnn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nnn_hash_table)
        new_excluded_asites[0] = trial_vac_asite[0]
        new_excluded_bsites[:len(nn_neighbors)] = nn_neighbors
        new_excluded_asites[1:len(nnn_neighbors) + 1] = nnn_neighbors

        exclusion_mask_bsites[
            [hash_table_bsites[val]
             for val in new_excluded_bsites[new_excluded_bsites != -1]]
        ] = False

        while_loop_counter += 1

        curr_rng = np.random.RandomState(seed=random_seeds[while_loop_counter])
        trial_vac_bsite = curr_rng.choice(bsites[exclusion_mask_bsites], size=1)

        more_nn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nn_hash_table)
        more_nnn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nnn_hash_table)
        new_excluded_bsites[len(nn_neighbors)] = trial_vac_bsite[0]
        new_excluded_asites[len(nnn_neighbors) + 1:len(nnn_neighbors) + 1 + len(more_nn_neighbors)] = more_nn_neighbors
        new_excluded_bsites[len(nn_neighbors) + 1:len(nn_neighbors) + 1 + len(more_nnn_neighbors)] = more_nnn_neighbors

        exclusion_mask_asites[
            [hash_table_asites[val]
             for val in new_excluded_asites[new_excluded_asites != -1]]
        ] = False
        exclusion_mask_bsites[
            [hash_table_bsites[val]
             for val in new_excluded_bsites[new_excluded_bsites != -1]]
        ] = False

        while_loop_counter += 1

        all_site_vacancies[while_loop_counter - 2] = trial_vac_asite[0]
        all_site_vacancies[while_loop_counter - 1] = trial_vac_bsite[0]

    if equal_vd_check:
        check_result = check_equal_sublattice_vd(all_site_vacancies, asites, bsites)
        print('Is equal vacancy density on each sublattice maintained: {}'.format(check_result))

    non_vacancy_inds = np.setdiff1d(np.arange(total_num_sites).astype(np.int64), all_site_vacancies)
    sparse_ham = sparse_ham.tocsr()
    sparse_ham = sparse_ham[non_vacancy_inds, :][:, non_vacancy_inds]

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham)
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))

    return sparse_ham


def hyperbolic_q3_equal_sublattice_vacancy_density_bulk_only_distance_restriction(pval, nval, vacancy_density,
                                                                                  seed, isocheck=False,
                                                                                  equal_vd_check=False):
    if (pval % 2) != 0:
        raise ValueError
    sparse_ham = general_hyperbolic_q3_hamiltonian(pval, nval)

    points_per_level, total_num_sites = number_points_q3_general_from_repeating_pattern(pval, nval)
    bulk_cutoff = int(np.sum(points_per_level[:-1]))
    num_bulk_sublat_sites = int(bulk_cutoff / 2)
    num_bulk_vacancies_per_sublat = int(round(vacancy_density * num_bulk_sublat_sites, 0))

    asites, bsites = sublattice_label_q3(pval, nval)
    bulk_asites = asites[asites < bulk_cutoff]
    bulk_bsites = bsites[bsites < bulk_cutoff]

    all_site_vacancies = np.repeat(-1.0, int(2*num_bulk_vacancies_per_sublat))
    rng_generator = np.random.default_rng(seed=seed)
    while_loop_counter = 0
    nn_rows, nn_cols = identify_nearest_neighbors(sparse_ham)
    nn_hash_table = convert_neighbors_list_to_hash_table(nn_rows, nn_cols)
    nnn_rows, nnn_cols = identify_next_nearest_neighbors(sparse_ham)
    nnn_hash_table = convert_neighbors_list_to_hash_table(nnn_rows, nnn_cols)
    exclusion_mask_asites = np.repeat(True, len(bulk_asites))
    exclusion_mask_bsites = np.repeat(True, len(bulk_bsites))
    new_excluded_asites = np.zeros(10, dtype=np.int64)
    new_excluded_bsites = np.zeros(10, dtype=np.int64)

    hash_table_asites = {v: i for (v, i) in zip(bulk_asites, np.arange(len(bulk_asites), dtype=np.int64))}
    hash_table_bsites = {v: i for (v, i) in zip(bulk_bsites, np.arange(len(bulk_bsites), dtype=np.int64))}

    while while_loop_counter < (2 * num_bulk_vacancies_per_sublat):
        trial_vac_asite = rng_generator.choice(bulk_asites[exclusion_mask_asites], size=1)

        new_excluded_asites[:] = -1
        new_excluded_bsites[:] = -1
        nn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nn_hash_table)
        nnn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nnn_hash_table)
        new_excluded_asites[0] = trial_vac_asite[0]
        new_excluded_bsites[:len(nn_neighbors)] = nn_neighbors
        new_excluded_asites[1:len(nnn_neighbors) + 1] = nnn_neighbors

        exclusion_mask_bsites[
            [hash_table_bsites[val]
             for val in new_excluded_bsites[np.logical_and(new_excluded_bsites != -1,
                                                           new_excluded_bsites < bulk_cutoff)]]
        ] = False

        while_loop_counter += 1

        trial_vac_bsite = rng_generator.choice(bulk_bsites[exclusion_mask_bsites], size=1)

        more_nn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nn_hash_table)
        more_nnn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nnn_hash_table)
        new_excluded_bsites[len(nn_neighbors)] = trial_vac_bsite[0]
        new_excluded_asites[len(nnn_neighbors) + 1:len(nnn_neighbors) + 1 + len(more_nn_neighbors)] = more_nn_neighbors
        new_excluded_bsites[len(nn_neighbors) + 1:len(nn_neighbors) + 1 + len(more_nnn_neighbors)] = more_nnn_neighbors

        exclusion_mask_asites[
            [hash_table_asites[val]
             for val in new_excluded_asites[np.logical_and(new_excluded_asites != -1,
                                                           new_excluded_asites < bulk_cutoff)]]
        ] = False
        exclusion_mask_bsites[
            [hash_table_bsites[val]
             for val in new_excluded_bsites[np.logical_and(new_excluded_bsites != -1,
                                                           new_excluded_bsites < bulk_cutoff)]]
        ] = False

        while_loop_counter += 1

        all_site_vacancies[while_loop_counter-2] = trial_vac_asite[0]
        all_site_vacancies[while_loop_counter-1] = trial_vac_bsite[0]

    if equal_vd_check:
        check_result = check_equal_sublattice_vd(all_site_vacancies, asites, bsites)
        print('Is equal vacancy density on each sublattice maintained: {}'.format(check_result))
        if not check_result:
            raise ValueError('Equal sublattice vacancy density was not maintained!')

    non_vacancy_inds = np.setdiff1d(np.arange(total_num_sites).astype(np.int64), all_site_vacancies)
    sparse_ham = sparse_ham.tocsr()
    sparse_ham = sparse_ham[non_vacancy_inds, :][:, non_vacancy_inds]

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham)
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))
        if check_res:
            raise ValueError('Some site(s) have had all neighbors removed!')

    return sparse_ham


def honeycomb_pbc_flake_equal_sublattice_vacancy_density_distance_restriction(nval, vacancy_density, seed,
                                                                              isocheck=False, equal_vd_check=False):
    sparse_ham = honeycomb_lattice_sparse_PBC(nval)

    total_num_sites = honeycomb_points(nval)[1]
    num_sublat_sites = int(total_num_sites / 2)
    num_vacancies_on_sublat = round(vacancy_density * num_sublat_sites, 0)

    asites, bsites = honeycomb_site_assignment(nval)

    all_site_vacancies = np.repeat(-1.0, int(2 * num_vacancies_on_sublat))
    rng_generator = np.random.default_rng(seed=seed)
    while_loop_counter = 0
    nn_rows, nn_cols = identify_nearest_neighbors(sparse_ham)
    nn_hash_table = convert_neighbors_list_to_hash_table(nn_rows, nn_cols)
    nnn_rows, nnn_cols = identify_next_nearest_neighbors(sparse_ham)
    nnn_hash_table = convert_neighbors_list_to_hash_table(nnn_rows, nnn_cols)
    exclusion_mask_asites = np.repeat(True, len(asites))
    exclusion_mask_bsites = np.repeat(True, len(bsites))
    new_excluded_asites = np.zeros(10, dtype=np.int64)
    new_excluded_bsites = np.zeros(10, dtype=np.int64)

    hash_table_asites = {v: i for (v, i) in zip(asites, np.arange(len(asites), dtype=np.int64))}
    hash_table_bsites = {v: i for (v, i) in zip(bsites, np.arange(len(bsites), dtype=np.int64))}

    while while_loop_counter < (2 * num_vacancies_on_sublat):
        trial_vac_asite = rng_generator.choice(asites[exclusion_mask_asites], size=1)

        new_excluded_asites[:] = -1
        new_excluded_bsites[:] = -1
        nn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nn_hash_table)
        nnn_neighbors = get_neighbors_of_site(trial_vac_asite[0], nnn_hash_table)
        new_excluded_asites[0] = trial_vac_asite[0]
        new_excluded_bsites[:len(nn_neighbors)] = nn_neighbors
        new_excluded_asites[1:len(nnn_neighbors) + 1] = nnn_neighbors

        exclusion_mask_bsites[
            [hash_table_bsites[val]
             for val in new_excluded_bsites[new_excluded_bsites != -1]]
        ] = False

        while_loop_counter += 1

        trial_vac_bsite = rng_generator.choice(bsites[exclusion_mask_bsites], size=1)

        more_nn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nn_hash_table)
        more_nnn_neighbors = get_neighbors_of_site(trial_vac_bsite[0], nnn_hash_table)
        new_excluded_bsites[len(nn_neighbors)] = trial_vac_bsite[0]
        new_excluded_asites[len(nnn_neighbors) + 1:len(nnn_neighbors) + 1 + len(more_nn_neighbors)] = more_nn_neighbors
        new_excluded_bsites[len(nn_neighbors) + 1:len(nn_neighbors) + 1 + len(more_nnn_neighbors)] = more_nnn_neighbors

        exclusion_mask_asites[
            [hash_table_asites[val]
             for val in new_excluded_asites[new_excluded_asites != -1]]
        ] = False
        exclusion_mask_bsites[
            [hash_table_bsites[val]
             for val in new_excluded_bsites[new_excluded_bsites != -1]]
        ] = False

        while_loop_counter += 1

        all_site_vacancies[while_loop_counter - 2] = trial_vac_asite[0]
        all_site_vacancies[while_loop_counter - 1] = trial_vac_bsite[0]

    if equal_vd_check:
        check_result = check_equal_sublattice_vd(all_site_vacancies, asites, bsites)
        print('Is equal vacancy density on each sublattice maintained: {}'.format(check_result))
        if not check_result:
            raise ValueError('Equal sublattice vacancy density was not maintained!')

    non_vacancy_inds = np.setdiff1d(np.arange(total_num_sites).astype(np.int64), all_site_vacancies)
    sparse_ham = sparse_ham.tocsr()
    sparse_ham = sparse_ham[non_vacancy_inds, :][:, non_vacancy_inds]

    if isocheck:
        check_res = check_for_isolated_sites(sparse_ham)
        print('Are there any sites that have had all neighbors removed: {}'.format(check_res))
        if check_res:
            raise ValueError('Some site(s) have had all neighbors removed!')

    return sparse_ham

