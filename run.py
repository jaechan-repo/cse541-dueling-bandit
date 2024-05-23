from algo import run_BTM, run_IF1, run_MultiSBM, run_RMED, run_RUCB, run_SAVAGE, run_Sparring
from algo.utils import get_compare_fn, get_regret_fn

import argparse
import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from pathlib import Path


def get_condorcet(P):
    return np.all(P >= 0.5, axis=1).nonzero()[0]


def run_bandit(algo, K, T, compare_fn, regret_fn) -> Tuple[int, NDArray]:
    match algo:
        case 'BTM': return run_BTM(K, T, compare_fn, regret_fn)
        case 'IF1': return run_IF1(K, T, compare_fn, regret_fn)
        case 'MultiSBM': return run_MultiSBM(K, T, compare_fn, regret_fn)
        case 'RMED': return run_RMED(K, T, compare_fn, regret_fn, mode=3)
        case 'RUCB': return run_RUCB(K, T, compare_fn, regret_fn)
        case 'SAVAGE': return run_SAVAGE(K, T, compare_fn, regret_fn)
        case 'Sparring': return run_Sparring(K, T, compare_fn, regret_fn)
        case _: raise ValueError("Algorithm not recognized.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', '--algos', nargs='+', default=[])
    parser.add_argument('-P', '--pref_file', type=str)
    parser.add_argument('--plot_title', type=str)
    parser.add_argument('--horizon', type=int, default=1_000_000)
    parser.add_argument('--n_repeats', type=int, default=10)
    args = parser.parse_args()

    if not args.pref_file.endswith('.npy'):
        raise ValueError("File to the preference matrix must have an extension '.npy'.")

    return args


if __name__ == '__main__':
    args = get_args()
    out = Path(args.pref_file).stem

    P = np.load(args.pref_file)
    K = P.shape[0]
    assert (P + P.T == 1).all(), "Preference matrix plus its transpose must equal 1.0 everywhere."

    condorcet = get_condorcet(P)
    assert len(condorcet) == 1, "There must be (only) one Condorcet winner."

    best_arm = condorcet[0]
    compare_fn = get_compare_fn(P)
    regret_fn = get_regret_fn(P, best_arm)

    regrets = []
    for algo in args.algos:
        regret_many_runs = []
        for _ in range(args.n_repeats):
            _, regret = run_bandit(algo, K, args.horizon, compare_fn, regret_fn)
            regret_many_runs.append(regret)
        regrets.append(np.array(regret_many_runs))

    np.save(f"./regrets/{out}.npy", np.array(regrets))
