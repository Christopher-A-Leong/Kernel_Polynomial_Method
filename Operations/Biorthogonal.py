import numpy as np

def biorthogonal_normalize(ls_mat,rs_mat):
    norm_factors = np.array([])
    for i in range(np.size(ls_mat, 0)):
        # print('i:{}'.format(i))
        norm_factors = np.append(norm_factors, np.sqrt(np.sum(ls_mat[:, i] * rs_mat[:, i]), dtype=np.clongdouble))

    ls_mat_norm_new = ls_mat/norm_factors
    rs_mat_norm_new = rs_mat/norm_factors

    return ls_mat_norm_new, rs_mat_norm_new
