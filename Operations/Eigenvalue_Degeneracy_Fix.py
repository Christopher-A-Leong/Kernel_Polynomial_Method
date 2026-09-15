import numpy as np

def small_disorder(chaos_order, num_sites):
    chaos = np.array([])
    for n in range(int(num_sites)):
        chaos = np.append(chaos, np.random.uniform(-chaos_order/2,chaos_order/2))
    chaos_mat = np.diag(chaos)
    chaos_correction_val = np.trace(chaos_mat)/num_sites
    chaos_correction_mat = np.eye(int(num_sites))*chaos_correction_val
    return chaos_mat - chaos_correction_mat


def small_disorder_specifyseed(chaos_order, num_sites, seed):
    random_generator = np.random.default_rng(seed)
    chaos = random_generator.uniform(-chaos_order/2, chaos_order/2, size=int(num_sites))
    chaos_mat = np.diag(chaos)
    chaos_correction_val = np.trace(chaos_mat)/num_sites
    chaos_correction_mat = np.eye(int(num_sites))*chaos_correction_val
    return chaos_mat - chaos_correction_mat

