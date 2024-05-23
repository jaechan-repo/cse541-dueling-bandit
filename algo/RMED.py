from .utils import DuelingBandit

import numpy as np

from typing import Tuple, Set, Callable, Literal
from numpy.typing import NDArray


def run_RMED(K: int, T: int,
             compare_fn: Callable[[int, int], Literal[0, 1]],
             regret_fn: Callable[[int, int], float],
             mode: Literal[1, 2, 3] = 1, kweight: float = 0.3, alpha1: float = 3.0
             ) -> Tuple[int, NDArray]:
    bandit = RMED(K, T, mode=mode, kweight=kweight, alpha1=alpha1)
    regrets = []

    for t in range(T):
        i, j = bandit.select_next_pair(t)
        res = compare_fn(i, j)
        bandit.update_state(i, j, res, t)
        regret = regret_fn(i, j)
        regrets.append(regret)

    return bandit.get_best_arm(), np.cumsum(regrets)



class RMED(DuelingBandit):

    def __init__(self, K: int, T: int,
                 mode: Literal[1, 2, 3] = 1,
                 kweight: float = 0.3, alpha1: float = 3.0):
        self.K = K
        self.c = 1
        self.T = T
        self.alpha1 = alpha1
        self.kweight = kweight
        self.RMED1 = (mode == 1)
        self.RMED2 = (mode == 2)
        self.RMED2Finite = (mode == 3)
        self.Tl = 0  # Initial exploration per arm
        self.res_matrix = np.zeros((K, K), dtype=int)
        self.LC = set()
        self.LRc = set()
        self.LN = set()
        self.first_round_in_main_phase = True
        self.bsi = []


    def empirical_winning_rate(self, i: int, j: int):
        if i == j:
            return 0.5
        denominator = self.res_matrix[i, j] + self.res_matrix[j, i]
        if denominator == 0:
            return 0.5
        return self.res_matrix[i, j] / denominator
    

    def most_losing_arm(self, i: int, opponents: Set[int]) -> int:
        cbest = -1
        worst_p = 1.0
        for j in opponents:
            if i == j:
                continue
            pij = self.empirical_winning_rate(i, j)
            if pij <= worst_p:
                worst_p = pij
                cbest = j
        return cbest
    

    def smallest_regret_arm(self, i_opt: int, i: int) -> int:
        cbest = i_opt
        smallest_v = 1e7
        for j in range(self.K):
            pij = self.empirical_winning_rate(i, j)
            if pij >= 0.5:
                continue
            p_istar_i = self.empirical_winning_rate(i_opt, i)
            p_istar_j = self.empirical_winning_rate(i_opt, j)
            v = max(p_istar_i + p_istar_j - 1.0, 0.01)
            denominator = self.kl(pij, 0.5)
            v /= denominator
            if v < smallest_v:
                smallest_v = v
                cbest = j
        return cbest
    

    def pair_selected_num(self, i: int, j: int) -> int:
        return self.res_matrix[i, j] + self.res_matrix[j, i]


    def log_likelihood(self, i: int) -> float:
        ll = 0.0
        for j in range(self.K):
            if i == j:
                continue
            mu_ij = self.empirical_winning_rate(i, j)
            if mu_ij < 0.5:
                ll -= self.pair_selected_num(i, j) * self.kl(mu_ij, 0.5)
        return ll
    

    def get_opponents(self, i: int) -> Set[int]:
        ops = set()
        for j in range(self.K):
            if i == j:
                continue
            pij = self.empirical_winning_rate(i, j)
            if pij <= 0.50001:
                ops.add(j)
        return ops
    

    def select_next_pair(self, t: int):
        for i in range(self.K):
            for j in range(i + 1, self.K):
                Nij = self.res_matrix[i, j] + self.res_matrix[j, i]
                if ((self.RMED1 and Nij == 0) or
                    (self.RMED2 and Nij < self.alpha1 * np.log(np.log(t + 1) + 0.01)) or
                    (self.RMED2Finite and Nij < self.alpha1 * np.log(np.log(self.T) + 0.01))):
                    return (i, j)
        
        lls = [self.log_likelihood(i) for i in range(self.K)]
        i_opt = np.argmax(lls)
        
        if self.first_round_in_main_phase and self.RMED2Finite:
            for i in range(self.K):
                self.bsi.append(self.smallest_regret_arm(i_opt, i))
            self.first_round_in_main_phase = False
        
        i = next(iter(self.LC))
        opponents = self.get_opponents(i)
        
        if self.RMED1:
            if i_opt in opponents:
                return (i, i_opt)
            elif opponents:
                return (i, self.most_losing_arm(i, opponents))
            else:
                return (i, i_opt)
        elif self.RMED2:
            bs_i = self.smallest_regret_arm(i_opt, i)
            Nib = self.pair_selected_num(i, bs_i)
            if i_opt in opponents and self.pair_selected_num(i, i_opt) <= Nib / np.log(np.log(t + 1) + 0.01):
                return (i, i_opt)
            if bs_i in opponents:
                return (i, bs_i)
            elif i_opt in opponents:
                return (i, i_opt)
            elif opponents:
                return (i, self.most_losing_arm(i, opponents))
            else:
                return (i, i_opt)
        elif self.RMED2Finite:
            bs_i = self.bsi[i]
            Nib = self.pair_selected_num(i, bs_i)
            if i_opt in opponents and self.pair_selected_num(i, i_opt) <= Nib / np.log(np.log(t + 1) + 0.01):
                return (i, i_opt)
            if bs_i in opponents:
                return (i, bs_i)
            elif i_opt in opponents:
                return (i, i_opt)
            elif opponents:
                return (i, self.most_losing_arm(i, opponents))
            else:
                return (i, i_opt)


    def update_state(self, i: int, j: int, res: int, t: int):
        if res == 1:
            self.res_matrix[i, j] += 1
        elif res == 0:
            self.res_matrix[j, i] += 1
        else:
            raise ValueError()

        self.LC.discard(i)
        self.LRc.add(i)

        lls = [self.log_likelihood(i) for i in range(self.K)]
        lls_opt = max(lls)

        for j in self.LRc:
            if lls_opt - lls[j] <= np.log(t + 1) + self.kweight * pow(self.K, 1.01):
                self.LN.add(j)

        if not self.LC:
            self.LC = self.LN
            self.LRc.clear()
            for i in range(self.K):
                if i not in self.LC:
                    self.LRc.add(i)
            self.LN.clear()


    def get_best_arm(self):
        total_wins = np.sum(self.res_matrix, axis=1)
        return np.argmax(total_wins)
    

    def kl(self, p: float, q: float) -> float:
        if p == 0 or p == 1:
            return np.inf
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
