import numpy as np
import scipy


def honeycomb_points(nl):
    def num_sites_higherlevel(lev):
        return ((lev-2)*2*6)+(3*6)
    numbers_perlevel = np.array([])
    if nl >= 1:
        numbers_perlevel = np.append(numbers_perlevel, 6)
    if nl >= 2:
        numbers_perlevel = np.append(numbers_perlevel, 18)
    if nl >= 3:
        for n in range(3, nl+1):
            numbers_perlevel = np.append(numbers_perlevel, num_sites_higherlevel(n))
    total_numpoints = np.sum(numbers_perlevel)
    return numbers_perlevel, total_numpoints


def honeycomb_first_site_on_gen(num_levels):
    points_per_level = honeycomb_points(num_levels)[0]
    return np.concatenate((np.array([0]), np.cumsum(points_per_level))).astype(int)


def sites_conn_to_next_gen(nl, first_sites_on_gen):
    num_sector_edge_sites = 2 * nl + 1
    if nl == 0:
        return np.arange(6)
    else:
        if (nl % 2) == 1:
            first_half_edge = np.arange(first_sites_on_gen[nl] + 1,
                                        first_sites_on_gen[nl] + 1 + (nl - 1) + 1, 2)
        else:
            first_half_edge = np.arange(first_sites_on_gen[nl],
                                        first_sites_on_gen[nl] + nl + 1, 2)
        last_half_edge = np.arange(first_sites_on_gen[nl + 1] - nl,
                                   first_sites_on_gen[nl + 1], 2)
        second_edge = np.arange(first_sites_on_gen[nl] + nl + 1,
                                first_sites_on_gen[nl] + nl + 1 + num_sector_edge_sites, 2)
        other_edges = np.copy(second_edge)
        for i in range(1, 5):
            other_edges = np.append(other_edges,
                                    second_edge + i * (first_sites_on_gen[nl + 1] - first_sites_on_gen[nl]) / 6)
        return np.concatenate((first_half_edge, other_edges, last_half_edge)).astype(int)


def sites_conn_to_prev_gen(nl, first_sites_on_gen):
    if nl == 0:
        raise ValueError
    else:
        if (nl % 2) == 1:
            first_half_edge = np.arange(first_sites_on_gen[nl],
                                        first_sites_on_gen[nl] + nl + 1, 2)
        else:
            first_half_edge = np.arange(first_sites_on_gen[nl] + 1,
                                        first_sites_on_gen[nl] + 1 + (nl - 1) + 1, 2)
        last_half_edge = np.arange(first_sites_on_gen[nl + 1] - nl + 1,
                                   first_sites_on_gen[nl + 1], 2)
        second_edge = np.arange(first_sites_on_gen[nl] + nl + 2,
                                first_sites_on_gen[nl] + nl + 2 + 2 * (nl - 1) + 1, 2)
        other_edges = np.copy(second_edge)
        for i in range(1, 5):
            other_edges = np.append(other_edges,
                                    second_edge + i * (first_sites_on_gen[nl + 1] - first_sites_on_gen[nl]) / 6)
        return np.concatenate((first_half_edge, other_edges, last_half_edge)).astype(int)


def honeycomb_lattice_sparse(num_levels):
    total_num_sites = int(honeycomb_points(num_levels)[1])
    ham = scipy.sparse.dok_matrix((total_num_sites, total_num_sites), dtype=np.int64)

    t = 1
    first_site_on_gen = honeycomb_first_site_on_gen(num_levels)
    for nl in range(num_levels - 1):

        # Intralayer Hopping
        sites_on_gen = np.arange(first_site_on_gen[nl], first_site_on_gen[nl + 1])
        ham[sites_on_gen, np.roll(sites_on_gen, -1)] = t
        ham[np.roll(sites_on_gen, -1), sites_on_gen] = t

        # Interlayer Hopping
        curr_gen_sites_conn_to_above = sites_conn_to_next_gen(nl, first_site_on_gen)
        next_gen_sites_conn_to_below = sites_conn_to_prev_gen(nl + 1, first_site_on_gen)
        ham[curr_gen_sites_conn_to_above, next_gen_sites_conn_to_below] = t
        ham[next_gen_sites_conn_to_below, curr_gen_sites_conn_to_above] = t

    # Last generation intralayer hopping
    sites_on_gen = np.arange(first_site_on_gen[num_levels - 1], first_site_on_gen[num_levels])
    ham[sites_on_gen, np.roll(sites_on_gen, -1)] = t
    ham[np.roll(sites_on_gen, -1), sites_on_gen] = t

    return ham


def add_periodic_boundary_sparse(nl, tbham_sparse):
    t = 1
    first_site_on_gen = honeycomb_first_site_on_gen(nl)
    pbc_conn_sites = sites_conn_to_next_gen(nl - 1, first_site_on_gen)

    if (nl % 2) == 1:
        if int((nl - 1) / 2) == ((nl - 1) / 2):
            pbc_conn_sites_edge_seg = np.split(np.roll(pbc_conn_sites, int((nl - 1) / 2)), 6)
        else:
            raise ValueError
    else:
        pbc_conn_sites_edge_seg = np.split(np.roll(pbc_conn_sites, int(nl / 2)), 6)

    tbham_sparse[pbc_conn_sites_edge_seg[0], np.flip(pbc_conn_sites_edge_seg[3])] = t
    tbham_sparse[np.flip(pbc_conn_sites_edge_seg[3]), pbc_conn_sites_edge_seg[0]] = t

    tbham_sparse[pbc_conn_sites_edge_seg[1], np.flip(pbc_conn_sites_edge_seg[4])] = t
    tbham_sparse[np.flip(pbc_conn_sites_edge_seg[4]), pbc_conn_sites_edge_seg[1]] = t

    tbham_sparse[pbc_conn_sites_edge_seg[2], np.flip(pbc_conn_sites_edge_seg[5])] = t
    tbham_sparse[np.flip(pbc_conn_sites_edge_seg[5]), pbc_conn_sites_edge_seg[2]] = t

    return tbham_sparse


def honeycomb_lattice_sparse_PBC(num_levels):
    tbham_sparse = honeycomb_lattice_sparse(num_levels)
    tbham_sparse = add_periodic_boundary_sparse(num_levels, tbham_sparse)
    return tbham_sparse


def ps_phase_intralayer_hopping(nl, beta, anchor_sites, prev_gen_ps_phases):
    num_sites_interval = anchor_sites[1:] - anchor_sites[:-1]
    if nl == 0:
        raise ValueError
    elif nl == 1:
        num_sites_interval = np.append(num_sites_interval, 3)
    else:
        num_sites_interval = np.append(num_sites_interval, 2)

    num_sites_interval = num_sites_interval.astype(int)
    prev_gen_sum_key = np.copy(num_sites_interval)
    prev_gen_sum_key[prev_gen_sum_key == 3] = 1

    if (nl % 2) == 0:
        new_ps_phases = np.add.reduceat(np.roll(prev_gen_ps_phases, -1),
                                        np.cumsum(np.append(np.array([0]), prev_gen_sum_key))[:-1])
        return np.roll(np.repeat((beta + new_ps_phases) / num_sites_interval, num_sites_interval), 1)

    else:
        new_ps_phases = np.add.reduceat(prev_gen_ps_phases,
                                        np.cumsum(np.append(np.array([0]), prev_gen_sum_key))[:-1])
        return np.repeat((beta + new_ps_phases) / num_sites_interval, num_sites_interval)


def honeycomb_real_magnetic_field_sparse(num_levels, beta):
    total_num_sites = int(honeycomb_points(num_levels)[1])
    ham = scipy.sparse.dok_matrix((total_num_sites, total_num_sites), dtype=np.complex128)

    t = 1
    first_site_on_gen = honeycomb_first_site_on_gen(num_levels)

    # Hard code first gen
    sites_on_gen = np.arange(6)
    curr_gen_conn_to_above = sites_conn_to_next_gen(0, first_site_on_gen)
    next_gen_conn_to_below = sites_conn_to_prev_gen(1, first_site_on_gen)

    ham[sites_on_gen, np.roll(sites_on_gen, -1)] = t * np.exp(1j * beta / 6)
    ham[np.roll(sites_on_gen, -1), sites_on_gen] = t * np.exp(-1j * beta / 6)

    ham[curr_gen_conn_to_above, next_gen_conn_to_below] = t
    ham[next_gen_conn_to_below, curr_gen_conn_to_above] = t

    # Variables to carryover to next iteration
    prev_gen_ps_phases = np.repeat(beta / 6, 6)

    for nl in range(1, num_levels - 1):

        # Interlayer Hopping
        curr_gen_sites_conn_to_above = sites_conn_to_next_gen(nl, first_site_on_gen)
        next_gen_sites_conn_to_below = sites_conn_to_prev_gen(nl + 1, first_site_on_gen)
        ham[curr_gen_sites_conn_to_above, next_gen_sites_conn_to_below] = t
        ham[next_gen_sites_conn_to_below, curr_gen_sites_conn_to_above] = t

        # Intralayer hopping modified for PS Phases
        curr_gen_sites_conn_to_below = sites_conn_to_prev_gen(nl, first_site_on_gen)
        sites_on_gen = np.arange(first_site_on_gen[nl], first_site_on_gen[nl + 1])
        new_ps_phases = ps_phase_intralayer_hopping(nl, beta, curr_gen_sites_conn_to_below, prev_gen_ps_phases)
        ham[sites_on_gen, np.roll(sites_on_gen, -1)] = t * np.exp(1j * new_ps_phases)
        ham[np.roll(sites_on_gen, -1), sites_on_gen] = t * np.exp(-1j * new_ps_phases)

        # Update variables to carryover to next iteration
        prev_gen_ps_phases = new_ps_phases

    # Intralayer hopping for last gen
    curr_gen_sites_conn_to_below = sites_conn_to_prev_gen(num_levels - 1, first_site_on_gen)
    sites_on_gen = np.arange(first_site_on_gen[num_levels - 1], total_num_sites)
    new_ps_phases = ps_phase_intralayer_hopping(num_levels - 1, beta, curr_gen_sites_conn_to_below, prev_gen_ps_phases)
    ham[sites_on_gen, np.roll(sites_on_gen, -1)] = t * np.exp(1j * new_ps_phases)
    ham[np.roll(sites_on_gen, -1), sites_on_gen] = t * np.exp(-1j * new_ps_phases)

    return ham


def strain_intralayer_rvals(nl, anchor_sites, anchor_site_rvals):
    num_interval_sites = anchor_sites[1:] - anchor_sites[:-1]
    rvals = np.array([])
    for i, nis in enumerate(num_interval_sites):
        if nis == 2:
            rvals = np.append(rvals, np.array([anchor_site_rvals[i], anchor_site_rvals[i] + 1]))
        elif nis == 3:
            rvals = np.append(rvals, np.array([anchor_site_rvals[i],
                                               anchor_site_rvals[i] + 1,
                                               anchor_site_rvals[i+1] + 1]))
        else:
            raise ValueError
    if (nl % 2) == 1:
        if nl == 1:
            rvals = np.append(rvals, np.array([anchor_site_rvals[-1],
                                               anchor_site_rvals[-1] + 1,
                                               anchor_site_rvals[0] + 1]))
        else:
            rvals = np.append(rvals, np.array([anchor_site_rvals[-1],
                                               anchor_site_rvals[-1] + 1]))
    else:
        rvals = np.append(np.array([anchor_site_rvals[0] + 1]), rvals)
        rvals = np.append(rvals, anchor_site_rvals[-1])

    return rvals


def site_assignment_honeycomb_gen(nl, first_site_on_gen):
    if (nl % 2) == 1:
        asites = np.arange(first_site_on_gen[nl] + 1, first_site_on_gen[nl + 1], 2, dtype=int)
        bsites = np.arange(first_site_on_gen[nl], first_site_on_gen[nl + 1], 2, dtype=int)
    else:
        asites = np.arange(first_site_on_gen[nl], first_site_on_gen[nl + 1], 2, dtype=int)
        bsites = np.arange(first_site_on_gen[nl] + 1, first_site_on_gen[nl + 1], 2, dtype=int)
    return asites, bsites


def honeycomb_axial_magnetic_field_sparse(num_levels, qax):
    total_num_sites = int(honeycomb_points(num_levels)[1])
    ham = scipy.sparse.dok_matrix((total_num_sites, total_num_sites), dtype=np.float64)

    t = 1
    first_site_on_gen = honeycomb_first_site_on_gen(num_levels)

    # Hard code first gen
    sites_on_gen = np.arange(6)
    curr_gen_conn_to_above = sites_conn_to_next_gen(0, first_site_on_gen)
    next_gen_conn_to_below = sites_conn_to_prev_gen(1, first_site_on_gen)

    ham[sites_on_gen, np.roll(sites_on_gen, -1)] = t
    ham[np.roll(sites_on_gen, -1), sites_on_gen] = t

    curr_gen_asites, curr_gen_bsites = site_assignment_honeycomb_gen(0, first_site_on_gen)
    next_gen_asites, next_gen_bsites = site_assignment_honeycomb_gen(1, first_site_on_gen)

    curr_gen_conn_to_above_asites = np.intersect1d(curr_gen_conn_to_above, curr_gen_asites)
    curr_gen_conn_to_above_bsites = np.intersect1d(curr_gen_conn_to_above, curr_gen_bsites)
    next_gen_conn_to_below_asites = np.intersect1d(next_gen_conn_to_below, next_gen_asites)
    next_gen_conn_to_below_bsites = np.intersect1d(next_gen_conn_to_below, next_gen_bsites)

    ham[curr_gen_conn_to_above_asites, next_gen_conn_to_below_bsites] = t * np.exp(-qax)
    ham[next_gen_conn_to_below_bsites, curr_gen_conn_to_above_asites] = t * np.exp(-qax)

    ham[curr_gen_conn_to_above_bsites, next_gen_conn_to_below_asites] = t * np.exp(qax)
    ham[next_gen_conn_to_below_asites, curr_gen_conn_to_above_bsites] = t * np.exp(qax)

    # Variables to carryover to next iteration
    anchor_site_rvals = np.repeat(1, 6)
    curr_gen_conn_to_below = next_gen_conn_to_below

    for nl in range(1, num_levels - 1):
        sites_on_gen = np.arange(first_site_on_gen[nl], first_site_on_gen[nl + 1])
        curr_gen_conn_to_above = sites_conn_to_next_gen(nl, first_site_on_gen)
        next_gen_conn_to_below = sites_conn_to_prev_gen(nl + 1, first_site_on_gen)

        curr_gen_rvals = strain_intralayer_rvals(nl, curr_gen_conn_to_below, anchor_site_rvals)
        curr_gen_asites, curr_gen_bsites = site_assignment_honeycomb_gen(nl, first_site_on_gen)
        curr_gen_asites, curr_gen_asites_ind = np.intersect1d(sites_on_gen, curr_gen_asites, return_indices=True)[:2]
        curr_gen_bsites, curr_gen_bsites_ind = np.intersect1d(sites_on_gen, curr_gen_bsites, return_indices=True)[:2]

        ham[curr_gen_asites, curr_gen_bsites] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                  - curr_gen_rvals[curr_gen_bsites_ind] ** 2))
        ham[curr_gen_bsites, curr_gen_asites] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                  - curr_gen_rvals[curr_gen_bsites_ind] ** 2))

        if (nl % 2) == 1:
            ham[curr_gen_asites, np.roll(curr_gen_bsites, -1)] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                                   - curr_gen_rvals[np.roll(curr_gen_bsites_ind, -1)] ** 2))
            ham[np.roll(curr_gen_bsites, -1), curr_gen_asites] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                                   - curr_gen_rvals[np.roll(curr_gen_bsites_ind, -1)] ** 2))
        else:
            ham[curr_gen_asites, np.roll(curr_gen_bsites, 1)] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                                   - curr_gen_rvals[np.roll(curr_gen_bsites_ind, 1)] ** 2))
            ham[np.roll(curr_gen_bsites, 1), curr_gen_asites] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                                   - curr_gen_rvals[np.roll(curr_gen_bsites_ind, 1)] ** 2))

        curr_gen_anchor_a, curr_gen_anchor_a_ind = np.intersect1d(curr_gen_conn_to_above,
                                                                  curr_gen_asites,
                                                                  return_indices=True)[:2]
        curr_gen_anchor_b, curr_gen_anchor_b_ind = np.intersect1d(curr_gen_conn_to_above,
                                                                  curr_gen_bsites,
                                                                  return_indices=True)[:2]

        next_gen_anchor_a = next_gen_conn_to_below[curr_gen_anchor_b_ind]
        next_gen_anchor_b = next_gen_conn_to_below[curr_gen_anchor_a_ind]

        curr_gen_anchor_a_ind = np.intersect1d(sites_on_gen, curr_gen_anchor_a, return_indices=True)[1]
        curr_gen_anchor_b_ind = np.intersect1d(sites_on_gen, curr_gen_anchor_b, return_indices=True)[1]

        ham[curr_gen_anchor_a, next_gen_anchor_b] = t * np.exp(qax * (curr_gen_rvals[curr_gen_anchor_a_ind] ** 2
                                                                      - (curr_gen_rvals[curr_gen_anchor_a_ind] + 1) ** 2))
        ham[next_gen_anchor_b, curr_gen_anchor_a] = t * np.exp(qax * (curr_gen_rvals[curr_gen_anchor_a_ind] ** 2
                                                                      - (curr_gen_rvals[curr_gen_anchor_a_ind] + 1) ** 2))

        ham[curr_gen_anchor_b, next_gen_anchor_a] = t * np.exp(qax * ((curr_gen_rvals[curr_gen_anchor_a_ind] + 1) ** 2
                                                                      - curr_gen_rvals[curr_gen_anchor_a_ind] ** 2))
        ham[next_gen_anchor_a, curr_gen_anchor_b] = t * np.exp(qax * ((curr_gen_rvals[curr_gen_anchor_a_ind] + 1) ** 2
                                                                      - curr_gen_rvals[curr_gen_anchor_a_ind] ** 2))

        anchor_site_rvals = curr_gen_rvals[np.intersect1d(sites_on_gen,
                                                          curr_gen_conn_to_above,
                                                          return_indices=True)[1]] + 1
        curr_gen_conn_to_below = next_gen_conn_to_below

    sites_on_gen = np.arange(first_site_on_gen[num_levels - 1], first_site_on_gen[num_levels])

    curr_gen_rvals = strain_intralayer_rvals(num_levels - 1, curr_gen_conn_to_below, anchor_site_rvals)
    curr_gen_asites, curr_gen_bsites = site_assignment_honeycomb_gen(num_levels - 1, first_site_on_gen)
    curr_gen_asites, curr_gen_asites_ind = np.intersect1d(sites_on_gen, curr_gen_asites, return_indices=True)[:2]
    curr_gen_bsites, curr_gen_bsites_ind = np.intersect1d(sites_on_gen, curr_gen_bsites, return_indices=True)[:2]

    ham[curr_gen_asites, curr_gen_bsites] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                              - curr_gen_rvals[curr_gen_bsites_ind] ** 2))
    ham[curr_gen_bsites, curr_gen_asites] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                              - curr_gen_rvals[curr_gen_bsites_ind] ** 2))

    if ((num_levels - 1) % 2) == 1:
        ham[curr_gen_asites, np.roll(curr_gen_bsites, -1)] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                               - curr_gen_rvals[np.roll(curr_gen_bsites_ind, -1)] ** 2))
        ham[np.roll(curr_gen_bsites, -1), curr_gen_asites] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                               - curr_gen_rvals[np.roll(curr_gen_bsites_ind, -1)] ** 2))
    else:
        ham[curr_gen_asites, np.roll(curr_gen_bsites, 1)] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                              - curr_gen_rvals[np.roll(curr_gen_bsites_ind, 1)] ** 2))
        ham[np.roll(curr_gen_bsites, 1), curr_gen_asites] = t * np.exp(qax * (curr_gen_rvals[curr_gen_asites_ind] ** 2
                                                                              - curr_gen_rvals[np.roll(curr_gen_bsites_ind, 1)] ** 2))

    return ham


def honeycomb_site_assignment(num_levels):
    asites = np.array([], dtype=np.int64)
    bsites = np.array([], dtype=np.int64)
    first_site_on_gen = honeycomb_first_site_on_gen(num_levels)
    for nl in range(num_levels):
        new_asites, new_bsites = site_assignment_honeycomb_gen(nl, first_site_on_gen)
        asites = np.append(asites, new_asites)
        bsites = np.append(bsites, new_bsites)
    return asites, bsites



def honeycomb_real_magnetic_field_sparse_nonHermitian(num_levels, beta, alpha_NH):
    ham = honeycomb_real_magnetic_field_sparse(num_levels, beta).tocsr()
    asites, bsites = honeycomb_site_assignment(num_levels)
    sbl_basis = np.concatenate((asites, bsites))
    ham = ham[:, sbl_basis][sbl_basis, :]
    nh_multiply_factor = np.concatenate((np.repeat(1 - alpha_NH, int(ham.shape[0]/2)),
                                         np.repeat(1 + alpha_NH, int(ham.shape[0]/2))))
    return ham.multiply(nh_multiply_factor)
