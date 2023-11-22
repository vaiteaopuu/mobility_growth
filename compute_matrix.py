from evo_four_states import evo_loop, MUTANTS, read_pheno, get_fitness, compute_traj_mut
from collections import Counter
import numpy as np
from numpy import mean, linspace, array, meshgrid
from multiprocessing import Pool
from evo_four_states import read_pheno, evo_loop, evo_loop_env_change


nb_el = 21
wl_l, w0_l, wk_l = np.linspace(0, 1, num=nb_el), np.linspace(-2, 0, num=nb_el), np.linspace(0, 4, num=nb_el)

glob_g = ["CRP", "Fis", "HNS"]
local_g = ["FlhDC", "FliZ"]
env_l = ["LB", "Glucose", "Lactose"]
pop_size = 1000
temp = 7
nb_gen = 1000
swap_freq = 100

for gen_a in glob_g:
    for gen_b in local_g:
        pheno_env = []
        for env in ["LB", "Glucose", "Lactose"]:
            pheno_env += [read_pheno(gen_a, gen_b, env)]
        def compute_counts(args):
            _, parms = args
            pop, traj, traj_mut, env_l, pop_l = evo_loop_env_change(pheno_env, nb_el=pop_size, nb_steps=nb_gen,
                                                                    parms=parms, temp=temp, swap_freq=swap_freq, env=1,
                                                                    save_pop=True, mut_rate=0.2)
            mut_traj_count = compute_traj_mut(traj_mut[5:])
            tot_counts = sum(c for el, c in mut_traj_count.items() if el is not None)
            if tot_counts > 0:
                mut_list = [el for el, c in mut_traj_count.items() if el is not None]
                mut_traj_count = {el: mut_traj_count[el]/tot_counts for el in mut_traj_count}
                counts_glob = sum(c for el, c in mut_traj_count.items() if el is not None and ((el[0][0] == -1 and el[1][0] == 1) or (el[0][0] == 1 and el[1][0] == -1)))
            else:
                counts_glob = 0
            return counts_glob

        # loop over x0
        parms_l = [((i, j, k), [wl, w0, wk]) for i, wl in enumerate(wl_l) for j, w0 in enumerate(w0_l) for k, wk in enumerate(wk_l)]
        pool = Pool(30)
        results = pool.map(compute_counts, parms_l)

        res_mat_g = np.zeros((nb_el, nb_el, nb_el))
        for (c_glob), ((i, j, k), parms) in zip(results, parms_l):
            res_mat_g[i, j, k] = c_glob
        with open(f"full/pop_{pop_size}_{gen_a}_{gen_b}_{temp}.txt", "wb") as out:
            np.save(out, res_mat_g)
