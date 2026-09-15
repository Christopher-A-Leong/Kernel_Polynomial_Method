import numpy as np
import matplotlib.pyplot as plt
from Lattice.Honeycomb_Sparse import honeycomb_points, honeycomb_lattice_sparse

def honeycomb_lattice(nl):
    ham = honeycomb_lattice_sparse(nl)
    return ham.toarray()

def honeycomb_flake_plot(num_levels):
    a1vec = np.array([np.sqrt(3) / 2, -1 / 2])
    a2vec = np.array([0, 1])
    b1vec = np.array([1 / (2 * np.sqrt(3)), 1 / 2])
    b2vec = np.array([1 / (2 * np.sqrt(3)), -1 / 2])
    b3vec = np.array([-1 / np.sqrt(3), 0])

    point_one = np.array([0, 1])
    point_two = point_one + b3vec
    point_three = point_two - b1vec
    point_four = point_three + b2vec
    point_five = point_four - b3vec
    point_six = point_five + b1vec

    points = np.array([point_one, point_two, point_three, point_four, point_five, point_six])

    points_per_level = honeycomb_points(num_levels)[0]

    def motif_even_one(points, level):
        for i in range(level - 1):
            if i % 2 == 0:
                points = np.vstack((points, points[-1] - b2vec))
            else:
                points = np.vstack((points, points[-1] + b3vec))
        return (points)

    def motif_odd_one(points, level):
        for i in range(level - 1):
            if i % 2 == 0:
                points = np.vstack((points, points[-1] + b3vec))
            else:
                points = np.vstack((points, points[-1] - b2vec))
        return (points)

    def motif_two(points, level):
        motif_length = level - 1
        for i in range(motif_length):
            points = np.vstack((points, points[-1] + b3vec))
            points = np.vstack((points, points[-1] - b1vec))
        points = np.vstack((points, points[-1] + b3vec))
        return (points)

    def motif_three(points, level):
        for i in range(level):
            points = np.vstack((points, points[-1] - b1vec))
            points = np.vstack((points, points[-1] + b2vec))
        return (points)

    def motif_four(points, level):
        for i in range(level - 1):
            points = np.vstack((points, points[-1] - b3vec))
            points = np.vstack((points, points[-1] + b2vec))
        return (points)

    def motif_five(points, level):
        for i in range(level):
            points = np.vstack((points, points[-1] - b3vec))
            points = np.vstack((points, points[-1] + b1vec))
        return (points)

    def motif_six(points, level):
        for i in range(level - 1):
            points = np.vstack((points, points[-1] - b2vec))
            points = np.vstack((points, points[-1] + b1vec))
        return (points)

    def motif_seven_odd(points, level):
        for i in range(int((level - 1) / 2)):
            points = np.vstack((points, points[-1] - b2vec))
            points = np.vstack((points, points[-1] + b3vec))
        return (points)

    def motif_seven_even(points, level):
        for i in range(int(level / 2 - 1)):
            points = np.vstack((points, points[-1] - b2vec))
            points = np.vstack((points, points[-1] + b3vec))
        points = np.vstack((points, points[-1] - b2vec))
        return (points)

    first_to_first_dist = b1vec
    last_first_point = point_one
    for n in range(2, num_levels + 1):
        points = np.vstack((points, last_first_point + first_to_first_dist))
        last_first_point = np.copy(points[-1])
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

    tbham = honeycomb_lattice(num_levels)
    conns = np.where(tbham != 0)

    # plt.plot([points[conns[0], 0], points[conns[1], 0]], [points[conns[0], 1], points[conns[1], 1]], color='black')
    # plt.scatter(points[:, 0], points[:, 1], color='black')
    # plt.show()

    return points, conns

