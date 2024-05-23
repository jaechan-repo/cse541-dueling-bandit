from .utils import DuelingBandit

from itertools import product
import numpy as np

from typing import Tuple, Callable, Literal
from numpy.typing import NDArray


def run_SAVAGE(K: int, T: int,
               compare_fn: Callable[[int, int], Literal[0, 1]],
               regret_fn: Callable[[int, int], float],
               delta: float | None = None
               ) -> Tuple[int, NDArray]:
    bandit = SAVAGE(K, T, delta=delta)
    regrets = []

    for t in range(T):
        i, j = bandit.select_next_pair(t)
        res = compare_fn(i, j)
        bandit.update_state(i, j, res, t)
        regret = regret_fn(i, j)
        regrets.append(regret)

    return bandit.get_best_arm(), np.cumsum(regrets)


class SAVAGE(DuelingBandit):

    def __init__(self, K: int, T: int, delta: float | None = None):
        if delta is None: delta = 1. / T
        self.delta = delta
        self.K = K
        self.T = T
        self.res_matrix = np.zeros((K, K), dtype=int)
        self.W = set(range(K))  # Active bandit arms

    
    def get_confidence_bound(self, sample_num: int) -> float:
        """Calculate the confidence bound for a given number of samples.

        Args:
            sample_num (int): Number of samples.

        Returns:
            float: Confidence bound.
        """
        return np.sqrt(np.log(self.K * (self.K - 1) * self.T / self.delta) / (2.0 * sample_num))


    def min_trial_pair(self) -> Tuple[int, int]:
        """Find the pair of arms with the minimum number of trials.

        Returns:
            Tuple[int, int]: Pair of arms.
        """
        if not self.W:
            raise ValueError("Winner already defined.")
        
        min_value = np.inf
        p = (-1, -1)

        W = tuple(self.W)
        for i, j in product(W, W):
            if i == j: continue
            n = self.res_matrix[i, j] + self.res_matrix[j, i]
            if n < min_value:
                min_value = n
                p = (i, j)

        if min_value == np.inf:
            raise ValueError("Minimum sample not found.")
        
        return p
    

    def select_next_pair(self, *_) -> Tuple[int]:
        if not self.W:
            raise ValueError("No arm left.")
        elif len(self.W) == 1:
            i = next(iter(self.W))
            return i, i
        return self.min_trial_pair()
    

    def update_state(self, i: int, j: int, res: Literal[0] | Literal[1], *_):
        if res == 1: self.res_matrix[i, j] += 1
        elif res == 0: self.res_matrix[j, i] += 1
        else: raise ValueError()

        remove_set = set()

        for i in self.W:
            for j in range(self.K):
                n = self.res_matrix[i, j] + self.res_matrix[j, i]
                if n == 0: continue
                p = self.res_matrix[i, j] / n
                c = self.get_confidence_bound(n)
                if p + c < 0.5:
                    remove_set.add(i)
                    break

        self.W -= remove_set


    def get_best_arm(self) -> int:
        return np.argmax(np.sum(self.res_matrix, axis=1))
