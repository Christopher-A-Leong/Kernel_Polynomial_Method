import numpy as np
from scipy.sparse import dok_array
from Operations.Neighbors import identify_next_nearest_neighbors, convert_neighbors_list_to_hash_table


def number_points_q4_numeric(p, num_levels):
    num_sites_per_gen = np.array([], dtype=np.int64)
    threeside_counter = 0
    twoside_counter = 0
    twoside_counter_old = 0
    threeside_counter_old = 0
    for n in range(1, num_levels+1):
        # print(n, threeside_counter, twoside_counter, threeside_counter_old, twoside_counter_old)
        if n == 1:
            num_sites_per_gen = np.append(num_sites_per_gen, p)
            threeside_counter = threeside_counter + 1
        else:
            # num_sites_per_gen = np.append(num_sites_per_gen, ((p-2)*threeside_counter + (p-3)*twoside_counter - 2*twoside_counter_old)*p)
            # twoside_counter_old = twoside_counter
            # threeside_counter_old = threeside_counter
            # threeside_counter = (p-3)*threeside_counter_old + (p-2)*twoside_counter_old
            # twoside_counter = threeside_counter_old - twoside_counter_old
            num_sites_per_gen = np.append(num_sites_per_gen, ((p - 2) * threeside_counter + (p - 3) * twoside_counter) * p)
            twoside_counter_old = twoside_counter
            threeside_counter_old = threeside_counter
            threeside_counter = (p - 3) * threeside_counter_old + (p - 4) * twoside_counter_old
            twoside_counter = threeside_counter_old - twoside_counter_old + 2*twoside_counter_old
    return num_sites_per_gen, np.sum(num_sites_per_gen)


def symbolic_motif_q4(pval, num_levels):
    three_side_replace = ['3side'] * (pval - 3) + ['2side']
    two_side_replace = ['3side'] * (pval - 4) + ['2side']
    if num_levels < 2:
        raise ValueError
    else:
        prev_num_three_sides_sector = None
        prev_num_two_sides_sector = None
        motif = []
        for nl in range(1, num_levels):
            if nl == 1:
                prev_num_three_sides_sector = 1
                prev_num_two_sides_sector = 0
                motif = ['3side']
            else:
                num_two_sides_sector = prev_num_three_sides_sector + prev_num_two_sides_sector
                num_three_sides_sector = ((pval - 3) * prev_num_three_sides_sector +
                                          (pval - 4) * prev_num_two_sides_sector)
                three_side_mask = np.where(np.array(motif) == '3side')[0]
                # two_side_mask = np.where(np.array(motif) == '2side')[0]
                for i in range(len(motif)):
                    if i in three_side_mask:
                        motif[i] = three_side_replace
                    else:
                        motif[i] = two_side_replace
                prev_num_three_sides_sector = num_three_sides_sector
                prev_num_two_sides_sector = num_two_sides_sector
                motif = [element for nest in motif for element in nest]
            # print(nl, motif)
    return np.tile(np.array(motif), pval)


def symbolic_motif_q4_sector(pval, num_levels):
    three_side_replace = ['3side'] * (pval - 3) + ['2side']
    two_side_replace = ['3side'] * (pval - 4) + ['2side']
    if num_levels < 2:
        raise ValueError
    else:
        prev_num_three_sides_sector = None
        prev_num_two_sides_sector = None
        motif = []
        for nl in range(1, num_levels):
            if nl == 1:
                prev_num_three_sides_sector = 1
                prev_num_two_sides_sector = 0
                motif = ['3side']
            else:
                num_two_sides_sector = prev_num_three_sides_sector + prev_num_two_sides_sector
                num_three_sides_sector = ((pval - 3) * prev_num_three_sides_sector +
                                          (pval - 4) * prev_num_two_sides_sector)
                three_side_mask = np.where(np.array(motif) == '3side')[0]
                # two_side_mask = np.where(np.array(motif) == '2side')[0]
                for i in range(len(motif)):
                    if i in three_side_mask:
                        motif[i] = three_side_replace
                    else:
                        motif[i] = two_side_replace
                prev_num_three_sides_sector = num_three_sides_sector
                prev_num_two_sides_sector = num_two_sides_sector
                motif = [element for nest in motif for element in nest]
            # print(nl, motif)
    return np.array(motif)


def num_points_q4_from_motif(pval, num_levels):
    points_per_level = np.array([], dtype=np.int64)
    for n in range(1, num_levels + 1):
        if n == 1:
            points_per_level = np.append(points_per_level, pval)
        else:
            motif = symbolic_motif_q4(pval, n)
            num_three_sides = len(np.where(motif == '3side')[0])
            num_two_sides = len(np.where(motif == '2side')[0])
            points_per_level = np.append(points_per_level,
                                         num_three_sides * (pval - 2) + num_two_sides * (pval - 3))
    return points_per_level, np.sum(points_per_level)


def sites_on_level_q4(pval, nl):
    points_per_level, total_num_points = number_points_q4_numeric(pval, nl)
    first_point = np.sum(points_per_level[:-1])
    num_points_current_level = points_per_level[-1]
    return np.arange(num_points_current_level) + first_point


def points_that_connect_to_above_q4(pval, nl):
    sites_connect_to_next_layer = sites_on_level_q4(pval, nl)
    # Roll needed since first point connected to first and last point on next layer
    return np.roll(np.repeat(sites_connect_to_next_layer, 2), -1)


def points_that_connect_to_below_q4(pval, nl):
    points_per_level, total_num_points = number_points_q4_numeric(pval, nl)
    first_point = np.sum(points_per_level[:-1])
    points_conn_to_prev_layer = np.array([0], dtype=np.int64)
    motif = symbolic_motif_q4_sector(pval, nl)

    # print(motif)
    # for i in range(len(motif)):
    #     if motif[i] == '3side':
    #         points_conn_to_prev_layer = np.append(points_conn_to_prev_layer, points_conn_to_prev_layer[-1] + pval - 3)
    #         if i != len(motif) - 1:
    #             points_conn_to_prev_layer = np.append(points_conn_to_prev_layer, points_conn_to_prev_layer[-1] + 1)
    #     else:
    #         points_conn_to_prev_layer = np.append(points_conn_to_prev_layer, points_conn_to_prev_layer[-1] + pval - 4)
    #         if i != len(motif) - 1:
    #             points_conn_to_prev_layer = np.append(points_conn_to_prev_layer, points_conn_to_prev_layer[-1] + 1)

    if len(motif) != 1:
        threeside_mask = (motif == '3side')
        twoside_mask = (motif == '2side')
        numeric_motif = np.zeros(len(motif))
        numeric_motif[threeside_mask] = pval - 3
        numeric_motif[twoside_mask] = pval - 4
        numeric_motif = np.repeat(numeric_motif, 2)
        numeric_motif[1::2] = 1
        numeric_motif = numeric_motif[:-1]
        points_conn_to_prev_layer = np.cumsum(np.append(points_conn_to_prev_layer, numeric_motif))
    else:
        if motif[0] == '3side':
            points_conn_to_prev_layer = np.append(points_conn_to_prev_layer, points_conn_to_prev_layer[-1] + pval - 3)
        else:
            # points_conn_to_prev_layer = np.append(points_conn_to_prev_layer, points_conn_to_prev_layer[-1] + pval - 4)
            raise ValueError

    sector_add_factor = points_per_level[-1] / pval
    full_points_conn_to_prev_layer = np.copy(points_conn_to_prev_layer)
    for a in range(1, pval):
        full_points_conn_to_prev_layer = np.append(full_points_conn_to_prev_layer,
                                                   (a * sector_add_factor) + points_conn_to_prev_layer)
    return full_points_conn_to_prev_layer + first_point


def intra_layer_connections_q4_from_motif(pval, nl, tbham):
    points_per_level, total_num_points = number_points_q4_numeric(pval, nl)
    first_point = np.sum(points_per_level[:-1])

    # sector_add_factor = points_per_level[-1] / pval
    # motif = symbolic_motif_q4_sector(pval, nl)
    # for m in motif:
    #     if m == '3side':
    #         for a in range(pval):
    #             intra_conns.append(np.arange(first_point, first_point + pval - 3 + 1) + a * sector_add_factor)
    #         first_point = first_point + pval - 3 + 1
    #     else:
    #         for a in range(pval):
    #             intra_conns.append(np.arange(first_point, first_point + pval - 4 + 1) + a * sector_add_factor)
    #         first_point = first_point + pval - 4 + 1

    motif = symbolic_motif_q4_sector(pval, nl)
    numeric_motif_starts = np.zeros(len(motif))
    threeside_mask = (motif == '3side')
    twoside_mask = (motif == '2side')
    numeric_motif_starts[threeside_mask] = pval - 3 + 1
    numeric_motif_starts[twoside_mask] = pval - 4 + 1
    plaquet_start_sites = np.array([first_point])
    plaquet_start_sites = np.cumsum(np.append(plaquet_start_sites, numeric_motif_starts))[:-1]

    threeside_starts = plaquet_start_sites[threeside_mask]
    twoside_starts = plaquet_start_sites[twoside_mask]
    sector_add_factor = points_per_level[-1] / pval
    threeside_starts_sector = np.copy(threeside_starts)
    twoside_starts_sector = np.copy(twoside_starts)
    for a in range(1, pval):
        threeside_starts = np.append(threeside_starts, threeside_starts_sector + (a * sector_add_factor))
        twoside_starts = np.append(twoside_starts, twoside_starts_sector + (a * sector_add_factor))

    threeside_sites = threeside_starts.reshape(-1, 1) + np.arange(pval - 3)
    twoside_sites = twoside_starts.reshape(-1, 1) + np.arange(pval - 4)

    tbham[threeside_sites.reshape(-1), threeside_sites.reshape(-1) + 1] = 1
    tbham[threeside_sites.reshape(-1) + 1, threeside_sites.reshape(-1)] = 1

    tbham[twoside_sites.reshape(-1), twoside_sites.reshape(-1) + 1] = 1
    tbham[twoside_sites.reshape(-1) + 1, twoside_sites.reshape(-1)] = 1

    return tbham


def general_hamiltonian_q4(pval, num_levels):

    t = 1

    points_per_level, total_num_points = number_points_q4_numeric(pval, num_levels)
    ham = dok_array((total_num_points, total_num_points), dtype=np.int64)

    if num_levels <= 0:
        raise ValueError
    else:
        for nl in range(1, num_levels):
            if nl == 1:
                sites_on_level = sites_on_level_q4(pval, nl)
                ham[sites_on_level, np.roll(sites_on_level, -1)] = t
                ham[np.roll(sites_on_level, -1), sites_on_level] = t

                above_gen_intersites = points_that_connect_to_below_q4(pval, nl + 1)
                current_gen_intersites = points_that_connect_to_above_q4(pval, nl)
                ham[current_gen_intersites, above_gen_intersites] = t
                ham[above_gen_intersites, current_gen_intersites] = t

            else:
                ham = intra_layer_connections_q4_from_motif(pval, nl, ham)

                above_gen_intersites = points_that_connect_to_below_q4(pval, nl + 1)
                current_gen_intersites = points_that_connect_to_above_q4(pval, nl)
                ham[current_gen_intersites, above_gen_intersites] = t
                ham[above_gen_intersites, current_gen_intersites] = t

        ham = intra_layer_connections_q4_from_motif(pval, num_levels, ham)

    ham = ham.tocsr()
    return ham


def get_each_gen_one_plaquet_q4(pval, num_levels):
    points_per_gen = number_points_q4_numeric(pval, num_levels)[0]
    if num_levels >= 1:
        plaquet_sites = np.arange(pval, dtype=np.int64)
    else:
        raise ValueError

    running_ind = 1
    for nl in range(1, num_levels):
        prev_gen_anchors = points_that_connect_to_above_q4(pval, nl)
        curr_gen_anchors = points_that_connect_to_below_q4(pval, nl + 1)
        if (points_per_gen[nl-1] % pval) == 0:
            pfold_rot_trans_prev_gen = int(points_per_gen[nl-1] / pval)
        else:
            raise ValueError
        if (points_per_gen[nl] % pval) == 0:
            pfold_rot_trans_curr_gen = int(points_per_gen[nl] / pval)
        else:
            raise ValueError

        plaquet_sites = np.vstack((plaquet_sites,
                                   np.concatenate((prev_gen_anchors[:2] + pfold_rot_trans_prev_gen * running_ind,
                                                   np.arange(curr_gen_anchors[0], curr_gen_anchors[1] + 1)
                                                   + pfold_rot_trans_curr_gen * running_ind))
                                   ))
        running_ind = running_ind + 1
        if running_ind > (pval - 1):
            running_ind = 0

    return plaquet_sites.astype(np.int64)


def sublattice_label_q4_small_system(pval, nval):
    tbham = general_hamiltonian_q4(pval, nval)
    tbham = tbham.toarray()
    asites = np.arange(0, pval, 2, dtype=np.int64)
    bsites = np.arange(1, pval, 2, dtype=np.int64)
    while len(asites) != int(tbham.shape[0] / 2) or len(bsites) != int(tbham.shape[0] / 2):
        for b in bsites:
            asites = np.append(asites, np.where(tbham[b, :] != 0)[0])
        for a in asites:
            bsites = np.append(bsites, np.where(tbham[a, :] != 0)[0])
        asites = np.unique(asites)
        bsites = np.unique(bsites)

    return asites, bsites


def sublattice_label_q4(pval, nval):
    tbham = general_hamiltonian_q4(pval, nval)
    nnn_hash_table = convert_neighbors_list_to_hash_table(*identify_next_nearest_neighbors(tbham))
    asites = np.arange(0, pval, 2).astype(np.int64)
    not_yet_checked_asites = asites.copy()
    checked_asites = np.array([], dtype=np.int64)
    bsites = np.arange(1, pval, 2).astype(np.int64)
    not_yet_checked_bsites = bsites.copy()
    checked_bsites = np.array([], dtype=np.int64)
    while (len(asites) != int(tbham.shape[0] / 2)) or (len(bsites) != int(tbham.shape[0] / 2)):
        for na in not_yet_checked_asites:
            asites = np.append(asites, nnn_hash_table[na])
        for nb in not_yet_checked_bsites:
            bsites = np.append(bsites, nnn_hash_table[nb])
        asites = np.unique(asites)
        bsites = np.unique(bsites)
        checked_asites = np.append(checked_asites, not_yet_checked_asites)
        checked_bsites = np.append(checked_bsites, not_yet_checked_bsites)
        not_yet_checked_asites = np.setdiff1d(asites, checked_asites)
        not_yet_checked_bsites = np.setdiff1d(bsites, checked_bsites)

    return asites, bsites
