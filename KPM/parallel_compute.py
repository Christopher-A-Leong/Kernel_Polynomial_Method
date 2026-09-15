from joblib import parallel_config, Parallel, delayed
from KPM.KPM import moments_ADOS_general
import numpy as np
from KPM.measure import rescale_disorder_unity_window, rescale_operator_unity_window
import scipy.sparse
import pickle
from Operations.Specific_Disorders import bond_disorder


def random_vectors_arr_generate(num_rand_vecs, vec_dim):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(vec_dim, num_rand_vecs))


def rescaled_onsite_disorder_generate(W, num_sites, eigval_min, eigval_max, epsilon=0.01):
    return rescale_disorder_unity_window(np.random.uniform(-W/2, W/2, size=num_sites),
                                         eigval_min,
                                         eigval_max,
                                         epsilon=epsilon)


def run_KPM_ADOS_parallel_disorder(n_jobs, sparse_ham, W_list, number_moments,
                                   num_rand_vecs, rand_vec_dim,
                                   save_dir, save_name, save_index_list):

    def run_routine(sparse_ham, number_moments, random_vecs_arr, save_dir, save_name, save_index):
        result = moments_ADOS_general(sparse_ham, number_moments, random_vecs_arr)
        np.save(save_dir + '/' + save_name + '_' + save_index, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (rescale_operator_unity_window(sparse_ham + scipy.sparse.diags(np.random.uniform(-W/2, W/2,
                                                                                             size=rand_vec_dim),
                                                                           format='csr'),
                                           eigval_min=-3.12, eigval_max=3.12
                                           ),
            number_moments,
            random_vectors_arr_generate(num_rand_vecs, rand_vec_dim),
            save_dir,
            save_name,
            save_index)
            for (W, save_index) in zip(W_list, save_index_list)
        )


def run_KPM_ADOS_parallel_vacancies_q3(n_jobs, eigval_min, eigval_max,
                                       pval, nval, vac_density_list, number_moments,
                                       num_rand_vecs, rand_vec_dim,
                                       save_dir, save_name, save_index_list,
                                       random_seeds, vacancy_hyperbolicq3_func):

    def run_routine(sparse_ham, nummoments, random_vecs_arr, savedir, savename, save_index):
        result = moments_ADOS_general(sparse_ham, nummoments, random_vecs_arr)
        np.save(savedir + '/' + savename + '_' + save_index, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (
                rescale_operator_unity_window(
                    vacancy_hyperbolicq3_func(pval, nval, vac_density, seed=randseed, isocheck=True, equal_vd_check=True),
                    eigval_min=eigval_min,
                    eigval_max=eigval_max
                ),
                number_moments,
                random_vectors_arr_generate(num_rand_vecs, rand_vec_dim),
                save_dir,
                save_name,
                save_index
            )
            for (vac_density, randseed, save_index) in zip(vac_density_list, random_seeds, save_index_list)
        )

    metadata_dict = {
        'pval': pval,
        'nval': nval,
        '(Vacancy density, random seed, save_index)': zip(vac_density_list, random_seeds, save_index_list),
        'Number random vectors': num_rand_vecs,
        'Number moments up to': number_moments,
        'Eigenvalue extrema': (eigval_min, eigval_max)
    }

    sf = open(save_dir + '/' + 'metadata.pkl', 'wb')
    pickle.dump(metadata_dict, sf)
    sf.close()


def run_KPM_ADOS_parallel_vacancies_q3_generate_hamiltonian_parallel(n_jobs, eigval_min, eigval_max,
                                                                     pval, nval, vac_density_list, number_moments,
                                                                     num_rand_vecs, rand_vec_dim,
                                                                     save_dir, save_name, save_index_list,
                                                                     random_seeds, vacancy_hyperbolicq3_func):

    def run_routine(p_value, n_value, vd, r_seed, ev_min, ev_max, nummoments, n_rvecs, rvec_dim, savedir, savename, save_index):

        sparse_ham = rescale_operator_unity_window(
            vacancy_hyperbolicq3_func(p_value, n_value, vd, seed=r_seed, isocheck=True, equal_vd_check=True),
            eigval_min=ev_min,
            eigval_max=ev_max
        )

        result = moments_ADOS_general(sparse_ham, nummoments, random_vectors_arr_generate(n_rvecs, rvec_dim))
        np.save(savedir + '/' + savename + '_' + save_index, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (
                pval,
                nval,
                vac_density,
                randseed,
                eigval_min,
                eigval_max,
                number_moments,
                num_rand_vecs,
                rand_vec_dim,
                save_dir,
                save_name,
                save_index
            )
            for (vac_density, randseed, save_index) in zip(vac_density_list, random_seeds, save_index_list)
        )

    metadata_dict = {
        'pval': pval,
        'nval': nval,
        '(Vacancy density, random seed, save_index)': zip(vac_density_list, random_seeds, save_index_list),
        'Number random vectors': num_rand_vecs,
        'Number moments up to': number_moments,
        'Eigenvalue extrema': (eigval_min, eigval_max)
    }

    sf = open(save_dir + '/' + 'metadata.pkl', 'wb')
    pickle.dump(metadata_dict, sf)
    sf.close()


def run_KPM_ADOS_parallel_vacancies_honeycomb(n_jobs, eigval_min, eigval_max,
                                              nval, vac_density_list, number_moments,
                                              num_rand_vecs, rand_vec_dim,
                                              save_dir, save_name, save_index_list,
                                              random_seeds, vacancy_honeycomb_func):

    def run_routine(sparse_ham, nummoments, random_vecs_arr, savedir, savename, save_index):
        result = moments_ADOS_general(sparse_ham, nummoments, random_vecs_arr)
        np.save(savedir + '/' + savename + '_' + save_index, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (
                rescale_operator_unity_window(
                    vacancy_honeycomb_func(nval, vac_density, seed=randseed, isocheck=True, equal_vd_check=True),
                    eigval_min=eigval_min,
                    eigval_max=eigval_max
                ),
                number_moments,
                random_vectors_arr_generate(num_rand_vecs, rand_vec_dim),
                save_dir,
                save_name,
                save_index
            )
            for (vac_density, randseed, save_index) in zip(vac_density_list, random_seeds, save_index_list)
        )

    metadata_dict = {
        'nval': nval,
        '(Vacancy density, random seed, save_index)': zip(vac_density_list, random_seeds, save_index_list),
        'Number random vectors': num_rand_vecs,
        'Number moments up to': number_moments,
        'Eigenvalue extrema': (eigval_min, eigval_max)
    }

    sf = open(save_dir + '/' + 'metadata.pkl', 'wb')
    pickle.dump(metadata_dict, sf)
    sf.close()


def run_KPM_ADOS_parallel_vacancies_honeycomb_generate_hamiltonian_parallel(n_jobs, eigval_min, eigval_max,
                                                                            nval, vac_density_list, number_moments,
                                                                            num_rand_vecs, rand_vec_dim,
                                                                            save_dir, save_name, save_index_list,
                                                                            random_seeds, vacancy_honeycomb_func):

    def run_routine(num_levels, vacancy_density, seed, ev_min, ev_max,
                    nummoments, random_vecs_arr, savedir, savename, save_index):
        sparse_ham = rescale_operator_unity_window(
            vacancy_honeycomb_func(num_levels, vacancy_density, seed=seed, isocheck=True),
            eigval_min=ev_min,
            eigval_max=ev_max
        )
        result = moments_ADOS_general(sparse_ham, nummoments, random_vecs_arr)
        np.save(savedir + '/' + savename + '_' + save_index, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (
                nval,
                vac_density,
                randseed,
                eigval_min,
                eigval_max,
                number_moments,
                random_vectors_arr_generate(num_rand_vecs, rand_vec_dim),
                save_dir,
                save_name,
                save_index
            )
            for (vac_density, randseed, save_index) in zip(vac_density_list, random_seeds, save_index_list)
        )

    metadata_dict = {
        'nval': nval,
        '(Vacancy density, random seed, save_index)': zip(vac_density_list, random_seeds, save_index_list),
        'Number random vectors': num_rand_vecs,
        'Number moments up to': number_moments,
        'Eigenvalue extrema': (eigval_min, eigval_max)
    }

    sf = open(save_dir + '/' + 'metadata.pkl', 'wb')
    pickle.dump(metadata_dict, sf)
    sf.close()


def run_KPM_ADOS_parallel_bond_disorder(n_jobs, sparse_ham, W_list, number_moments,
                                        num_rand_vecs, rand_vec_dim, eigval_min, eigval_max,
                                        save_dir, save_name, save_index_list, seed_list):

    def run_routine(original_sparse_ham, Wb, disorder_seed, num_moments, number_rand_vecs, random_vec_dim,
                    savedir, savename, saveindex):

        result = moments_ADOS_general(rescale_operator_unity_window(original_sparse_ham
                                                                    + bond_disorder(Wb, original_sparse_ham, disorder_seed),
                                                                    eigval_min=eigval_min, eigval_max=eigval_max),
                                      num_moments,
                                      random_vectors_arr_generate(number_rand_vecs, random_vec_dim))
        np.save(savedir + '/' + savename + '_' + saveindex, result)

    with parallel_config(backend='loky'):
        Parallel(n_jobs=n_jobs)(
            delayed(run_routine)
            (sparse_ham,
             W,
             spec_seed,
             number_moments,
             num_rand_vecs,
             rand_vec_dim,
             save_dir,
             save_name,
             save_index)
            for (W, save_index, spec_seed) in zip(W_list, save_index_list, seed_list)
        )

    metadata_dict = {
        '(Wb, random seed, save_index)': zip(W_list, seed_list, save_index_list),
        'Number random vectors': num_rand_vecs,
        'Number moments up to': number_moments,
        'Eigenvalue extrema': (eigval_min, eigval_max)
    }

    sf = open(save_dir + '/' + 'metadata.pkl', 'wb')
    pickle.dump(metadata_dict, sf)
    sf.close()


def generate_random_seeds_for_parallel_jobs(original_seed, number_of_jobs):
    seeds_generator = np.random.SeedSequence(original_seed)
    seeds = seeds_generator.spawn(number_of_jobs)
    return seeds
