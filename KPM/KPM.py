import numpy as np
import scipy
import time


# def chebyshev_polynomial_vector_recursion_calc(rescaled_op, prevprev_op_vector_prod, prev_op_vector_prod):
#     result = 2 * rescaled_op @ prev_op_vector_prod - prevprev_op_vector_prod
#     return np.asarray(result), np.asarray(prev_op_vector_prod)


def chebyshev_polynomial_vector_recursion_calc_sharedham(rescaled_op, rescaled_disorder_vals,
                                                         prevprev_op_vector_prod, prev_op_vector_prod):
    result = (2 * np.asarray(rescaled_op @ prev_op_vector_prod)
              + 2 * (rescaled_disorder_vals.reshape(-1, 1) * prev_op_vector_prod)
              - prevprev_op_vector_prod)
    return np.asarray(result), np.asarray(prev_op_vector_prod)


def moments_ADOS_sharedham(sparse_rescaled_ham, rescaled_onsite_disorder_vals, number_moments, random_vectors_arr):
    moments = np.zeros(number_moments, dtype=np.float64)
    prev_op_vec_prod = None
    prevprev_op_vec_prod = None

    # Hard code 0th moment calculation
    if number_moments >= 0:
        prevprev_op_vec_prod = random_vectors_arr
        moments[0] = np.sum(random_vectors_arr.conj() * random_vectors_arr) / random_vectors_arr.shape[1]

    # Hard code 1st moment calculation
    if number_moments >= 1:
        prev_op_vec_prod = (np.asarray(sparse_rescaled_ham @ random_vectors_arr) +
                            (rescaled_onsite_disorder_vals.reshape(-1, 1) * random_vectors_arr))
        moments[1] = np.sum(random_vectors_arr.conj() * prev_op_vec_prod) / random_vectors_arr.shape[1]

    if number_moments >= 2:
        for nm in range(2, number_moments):
            # print(nm)
            prev_op_vec_prod, prevprev_op_vec_prod = chebyshev_polynomial_vector_recursion_calc_sharedham(
                sparse_rescaled_ham,
                rescaled_onsite_disorder_vals,
                prevprev_op_vec_prod,
                prev_op_vec_prod
            )

            moments[nm] = np.sum(random_vectors_arr.conj() * prev_op_vec_prod) / random_vectors_arr.shape[1]

    return moments / random_vectors_arr.shape[0]


def chebyshev_polynomial_vector_recursion_calc_general(rescaled_op, prevprev_op_vector_prod, prev_op_vector_prod):
    result = 2 * np.asarray(rescaled_op @ prev_op_vector_prod) - prevprev_op_vector_prod
    return np.asarray(result), np.asarray(prev_op_vector_prod)


def moments_ADOS_general(sparse_rescaled_ham, number_moments, random_vectors_arr):
    moments = np.zeros(number_moments, dtype=np.float64)
    prev_op_vec_prod = None
    prevprev_op_vec_prod = None

    st = time.time()

    # Hard code 0th moment calculation
    if number_moments >= 0:
        prevprev_op_vec_prod = random_vectors_arr
        moments[0] = np.sum((random_vectors_arr.conj() * random_vectors_arr) / (random_vectors_arr.shape[1] * random_vectors_arr.shape[0]))

    # Hard code 1st moment calculation
    if number_moments >= 1:
        prev_op_vec_prod = np.asarray(sparse_rescaled_ham @ random_vectors_arr)
        moments[1] = np.sum(np.real(random_vectors_arr.conj() * prev_op_vec_prod) / (random_vectors_arr.shape[1] * random_vectors_arr.shape[0]))

    if number_moments >= 2:
        for nm in range(2, number_moments):
            if (nm + 1) % 512 == 0:
                print('{} moments calculated in {} seconds'.format(nm + 1, time.time()-st))
                st = time.time()

            prev_op_vec_prod, prevprev_op_vec_prod = chebyshev_polynomial_vector_recursion_calc_general(
                sparse_rescaled_ham,
                prevprev_op_vec_prod,
                prev_op_vec_prod
            )

            moments[nm] = np.sum(np.real(random_vectors_arr.conj() * prev_op_vec_prod) / (random_vectors_arr.shape[1] * random_vectors_arr.shape[0]))

    return moments


def moments_LDOS_general(sparse_rescaled_ham, number_moments, local_site_indices):
    if (number_moments % 2) != 0:
        raise ValueError('Even number of moments expected')

    num_sites = sparse_rescaled_ham.shape[0]
    local_site_placeholder = np.arange(0, len(local_site_indices), dtype=int)
    moments = np.zeros((number_moments, len(local_site_indices)), dtype=np.float64)
    prev_op_vec_prod = None
    prevprev_op_vec_prod = None

    st = time.time()

    # Hard code 0th moment calculation
    if number_moments >= 0:
        prevprev_op_vec_prod = np.zeros((sparse_rescaled_ham.shape[0], len(local_site_indices)), dtype=np.float64)
        prevprev_op_vec_prod[local_site_indices, local_site_placeholder] = 1
        moments[0, :] = 1 / num_sites

    # Hard code 1st moment calculation
    if number_moments >= 1:
        prev_op_vec_prod = sparse_rescaled_ham[:, local_site_indices].toarray()
        moments[1, :] = prev_op_vec_prod[local_site_indices, local_site_placeholder] / num_sites

    calculated_moments = np.array([0, 1])
    if number_moments >= 2:
        for nm in range(2, int(np.floor(number_moments/2))):
            if (nm + 1) % 64 == 0:
                calculated_moments = np.unique(calculated_moments)
                print('{} moments calculated in {} seconds'.format(len(calculated_moments), time.time()-st))

            prev_op_vec_prod, prevprev_op_vec_prod = chebyshev_polynomial_vector_recursion_calc_general(
                sparse_rescaled_ham,
                prevprev_op_vec_prod,
                prev_op_vec_prod
            )
            if nm in calculated_moments:
                pass
            else:
                moments[nm, :] = prev_op_vec_prod[local_site_indices, local_site_placeholder] / num_sites
            moments[int(2 * nm), :] = (2 * np.sum(prev_op_vec_prod.conj() * prev_op_vec_prod / num_sites, axis=0)
                                       - moments[0, :])
            moments[int(2 * nm - 1), :] = (2 * np.sum(prev_op_vec_prod.conj() * prevprev_op_vec_prod / num_sites, axis=0)
                                           - moments[1, :])
            calculated_moments = np.append(calculated_moments, np.array([nm, int(2 * nm), int(2 * nm - 1)]))

        # Handles last moment which remains to be calculated after going through above for loop
        prev_op_vec_prod, prevprev_op_vec_prod = chebyshev_polynomial_vector_recursion_calc_general(
            sparse_rescaled_ham,
            prevprev_op_vec_prod,
            prev_op_vec_prod
        )

        moments[int(2 * int(np.floor(number_moments / 2)) - 1), :] = (2 * np.sum(prev_op_vec_prod
                                                                                 * prevprev_op_vec_prod
                                                                                 / num_sites, axis=0) - moments[1, :])

    return moments

