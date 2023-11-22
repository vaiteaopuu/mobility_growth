"""-1, -1 = Wt
-----
glob = 1
loc = 2

-----
Growth = 1
Swim = 2
"""

import numpy as np
# import matplotlib.pyplot as plt
from collections import Counter


MUTANTS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

def read_pheno(gene_a, gene_b, env):
    tmp = {}
    tmp[(-1, -1)] = [0., 0.]
    for l in open("./data/growth.data"):
        if not l.startswith("#"):
            val = l.strip().split()
            if env == val[1]:
                if val[0] == gene_a:
                    tmp[(1, -1)] = [float(val[2]), float(val[4])]
                elif val[0] == gene_b:
                    tmp[(-1, 1)] = [float(val[2]), float(val[4])]
                elif val[0] == gene_a+"+"+gene_b or val[0] == gene_b+"+"+gene_a:
                    tmp[(1, 1)] = [float(val[2]), float(val[4])]
                else:
                    "nope"
    return tmp


def norm_growth(pheno_dic):
    only_gr = [gr for gr, sw in pheno_dic.values()]
    range_v = max(only_gr)-min(only_gr)
    tmp = {k: [(gr-min(only_gr))/range_v, sw] for k, (gr, sw) in pheno_dic.items()}
    return tmp


def mutate_pop(pop, frac_mut=0.1):
    "flip ind"
    num_row = pop.shape[0]
    mutate = np.random.randint(0, num_row)
    tmp_pop = np.copy(pop)
    store_mut = [None for _ in range(num_row)]
    for i in range(num_row):
        if np.random.uniform(0, 1) < frac_mut:
            random_col_index = np.random.randint(0, 2)
            ref = tuple(tmp_pop[i, :])
            tmp_pop[i, random_col_index] *= -1
            mutated = tuple(tmp_pop[i, :])
            store_mut[i] = (ref, mutated)
    # print(store_mut)
    return tmp_pop, store_mut


def get_pheno(pop, pheno_dic):
    "lookup table for phenotypes"
    tmp = []
    for el in pop:
        tmp += [pheno_dic[tuple(el)]]
    return np.array(tmp)


def sigmoid(z, other=0):
    return 1/(1 + np.exp(-z+other))


def get_fitness(pop, pheno_dic, parms):
    pheno_pop = get_pheno(pop, pheno_dic)
    fit = parms[0] * pheno_pop[:, 0] + (1.-parms[0]) * sigmoid(parms[2] * (pheno_pop[:, 1]-parms[1]))
    return fit


def evo_loop_env_change(pheno_env=None, nb_el=10, nb_steps=100,
                        parms=[0.5, 0, 1.],
                        temp=2,
                        swap_freq=100,
                        env=None,
                        save_coords=False,
                        mut_rate=0.1,
                        save_pop=False,
                        burn_out=0):
    pop = np.random.choice([-1, 1], size=(nb_el, 2))
    if env is None:
        env = np.random.choice(range(len(pheno_env)))
    pop_size = pop.shape[0]
    tmp_fit, traj_mut, traj_mutb = [], [], []
    pop_l, pop_lb, env_l, env_lb = [], [], [], []
    bi = 0
    for i in range(nb_steps):
        new_pop, store_mut = mutate_pop(pop, mut_rate)
        fitness = get_fitness(new_pop, pheno_env[env], parms)
        wei_f = np.exp(fitness*temp)
        prob = wei_f/wei_f.sum()

        new_id = np.random.choice(range(pop_size), size=pop_size, p=prob, replace=True)
        pop = new_pop[new_id, :]
        tmp_fit += [fitness[new_id].mean()]

        if bi >= burn_out:
            traj_mut += [[store_mut[ni] for ni in new_id]]
            pop_l += [np.copy(pop)]
            env_l += [env]
        else:
            traj_mutb += [[store_mut[ni] for ni in new_id]]
            pop_lb += [np.copy(pop)]
            env_lb += [env]
        bi += 1
        if save_coords:
            pop_l += [get_pheno(pop, pheno_env[env])]
        if (i+1) % swap_freq == 0:
            env = np.random.choice(range(len(pheno_env)))
            bi = 0
    if save_coords or save_pop:
        # return pop, tmp_fit, traj_mut, env_l, pop_l, traj_mutb, pop_lb, env_lb
        return pop, tmp_fit, traj_mut, env_l, pop_l
    else:
        return pop, tmp_fit, traj_mut, env_l
