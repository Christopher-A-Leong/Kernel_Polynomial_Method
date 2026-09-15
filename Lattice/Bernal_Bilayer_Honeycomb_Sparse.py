import numpy as np
import scipy
from Lattice.Honeycomb_Sparse import honeycomb_lattice_sparse_PBC
from Lattice.Honeycomb_Sparse import honeycomb_lattice_sparse
from Lattice.Honeycomb_Sparse import honeycomb_points
from Lattice.Honeycomb_Sparse import sites_conn_to_next_gen
from Lattice.Honeycomb_Sparse import honeycomb_first_site_on_gen
from Lattice.Honeycomb_Sparse import honeycomb_site_assignment


def honeycomb_flake_plot(num_levels):
    # Assumed side length of fundamental hexagon is 1/sqrt(3)
    b1vec = np.array([1 / (2 * np.sqrt(3)), 1 / 2])
    b2vec = np.array([1 / (2 * np.sqrt(3)), -1 / 2])
    b3vec = np.array([-1 / np.sqrt(3), 0])

    # Establish center plaquette
    point_one = np.array([0, 1])
    point_two = point_one + b3vec
    point_three = point_two - b1vec
    point_four = point_three + b2vec
    point_five = point_four - b3vec
    point_six = point_five + b1vec

    points = np.array([point_one, point_two, point_three, point_four, point_five, point_six])

    # top right flake edge (first few sites on the generation) for even numbered generation
    def motif_even_one(ps, level):
        for i in range(level - 1):
            if i % 2 == 0:
                ps = np.vstack((ps, ps[-1] - b2vec))
            else:
                ps = np.vstack((ps, ps[-1] + b3vec))
        return ps

    # top right flake edge (first few sites on the generation) for odd numbered generation
    def motif_odd_one(ps, level):
        for i in range(level - 1):
            if i % 2 == 0:
                ps = np.vstack((ps, ps[-1] + b3vec))
            else:
                ps = np.vstack((ps, ps[-1] - b2vec))
        return ps

    # top left flake edge
    def motif_two(ps, level):
        motif_length = level - 1
        for i in range(motif_length):
            ps = np.vstack((ps, ps[-1] + b3vec))
            ps = np.vstack((ps, ps[-1] - b1vec))
        ps = np.vstack((ps, ps[-1] + b3vec))
        return ps

    # left flake edge
    def motif_three(ps, level):
        for i in range(level):
            ps = np.vstack((ps, ps[-1] - b1vec))
            ps = np.vstack((ps, ps[-1] + b2vec))
        return ps

    # bottom left flake edge
    def motif_four(ps, level):
        for i in range(level - 1):
            ps = np.vstack((ps, ps[-1] - b3vec))
            ps = np.vstack((ps, ps[-1] + b2vec))
        return ps

    # bottom right flake edge
    def motif_five(ps, level):
        for i in range(level):
            ps = np.vstack((ps, ps[-1] - b3vec))
            ps = np.vstack((ps, ps[-1] + b1vec))
        return ps

    # right flake edge
    def motif_six(ps, level):
        for i in range(level - 1):
            ps = np.vstack((ps, ps[-1] - b2vec))
            ps = np.vstack((ps, ps[-1] + b1vec))
        return ps

    # top right flake edge (last few sites on generation), odd numbered generation
    def motif_seven_odd(ps, level):
        for i in range(int((level - 1) / 2)):
            ps = np.vstack((ps, ps[-1] - b2vec))
            ps = np.vstack((ps, ps[-1] + b3vec))
        return ps

    # top right flake edge (last few sites on generation), even numbered generation
    def motif_seven_even(ps, level):
        for i in range(int(level / 2 - 1)):
            ps = np.vstack((ps, ps[-1] - b2vec))
            ps = np.vstack((ps, ps[-1] + b3vec))
        ps = np.vstack((ps, ps[-1] - b2vec))
        return ps

    first_to_first_dist = b1vec
    last_first_point = point_one
    for n in range(2, num_levels + 1):
        # Establish first point on new generation
        points = np.vstack((points, last_first_point + first_to_first_dist))
        last_first_point = np.copy(points[-1])
        # Handle rest of the sites on the generation
        if n % 2 == 0:
            first_to_first_dist = -b3vec + b1vec - b2vec
            points = motif_even_one(points, n)
            points = motif_two(points, n)
            points = motif_three(points, n)
            points = motif_four(points, n)
            points = motif_five(points, n)
            points = motif_six(points, n)
            points = motif_seven_even(points, n)
        else:
            first_to_first_dist = b1vec
            points = motif_odd_one(points, n)
            points = motif_two(points, n)
            points = motif_three(points, n)
            points = motif_four(points, n)
            points = motif_five(points, n)
            points = motif_six(points, n)
            points = motif_seven_odd(points, n)

    # get connectivity of lattice from OBC hamiltonian (not PBC to avoid PBC connections which we do not want to plot)
    tbham = honeycomb_lattice_sparse(num_levels)
    conns = tbham.nonzero()

    # Example of how to plot sites and bonds
    # plt.plot([points[conns[0], 0], points[conns[1], 0]], [points[conns[0], 1], points[conns[1], 1]], color='black')
    # plt.scatter(points[:, 0], points[:, 1], color='black')
    # plt.show()

    return points, conns


def bernal_bilayer_honeycomb_sparse_PBC(num_levels, overlap_floating_point_error=10**(-10)):
    # Establish hamiltonian (no interlayer hoppings yet)
    single_layer = honeycomb_lattice_sparse_PBC(num_levels)
    tbham = scipy.sparse.kron(np.array([[1, 0], [0, 1]]), single_layer)

    # Add interlayer hopping through graphical means
    locations, connections = honeycomb_flake_plot(num_levels)
    other_layer_locations = locations.copy()
    other_layer_locations[:, 0] = other_layer_locations[:, 0] - 1 / (2 * np.sqrt(3))
    other_layer_locations[:, 1] = other_layer_locations[:, 1] + 1 / 2

    comparison = locations - other_layer_locations.reshape(other_layer_locations.shape[0], 1, 2)
    comparison = np.abs(comparison) < overlap_floating_point_error
    extra_other, extra = np.where(np.logical_and(comparison[:, :, 0], comparison[:, :, 1]))

    # Need to also account for last bit of boundary interlayer hoppings due to PBC
    first_site_on_gen = honeycomb_first_site_on_gen(num_levels)
    last_gen_pbc_inds = sites_conn_to_next_gen(num_levels - 1, first_site_on_gen)
    num_sites_last_gen = honeycomb_points(num_levels)[0][-1]

    rel_edge_mask = np.logical_and((last_gen_pbc_inds - first_site_on_gen[-2])
                                   >= (num_levels + 3 * num_sites_last_gen / 6),
                                   (last_gen_pbc_inds - first_site_on_gen[-2])
                                   < (num_levels + 4 * num_sites_last_gen / 6))
    rel_edge_mask_other = np.logical_and((last_gen_pbc_inds - first_site_on_gen[-2])
                                         >= num_levels,
                                         (last_gen_pbc_inds - first_site_on_gen[-2])
                                         < (num_levels + 1 * num_sites_last_gen / 6))

    extra = np.append(extra, last_gen_pbc_inds[rel_edge_mask])
    extra_other = np.append(extra_other, np.flip(last_gen_pbc_inds[rel_edge_mask_other])) + single_layer.shape[0]

    # add interlayer hoppings
    add_term_x = np.concat((extra, extra_other))
    add_term_y = np.concat((extra_other, extra))

    interlayer_add_term = scipy.sparse.csr_array((np.ones(len(add_term_x)),
                                                  (add_term_x, add_term_y)),
                                                 shape=(tbham.shape[0], tbham.shape[1]))

    return tbham + interlayer_add_term


def sublattice_bernal_bilayer_honeycomb_label(num_levels):
    num_sites_single_layer = honeycomb_points(num_levels)[1]
    single_layer_asites, single_layer_bsites = honeycomb_site_assignment(num_levels)
    return (single_layer_asites, single_layer_bsites,
            single_layer_asites + num_sites_single_layer, single_layer_bsites + num_sites_single_layer)

