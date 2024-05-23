from .utils import DuelingBandit
from .MAB import MAB

import numpy as np
import random

from typing import Literal, Tuple, Callable
from numpy.typing import NDArray


def run_MultiSBM(K: int, T: int,
                 compare_fn: Callable[[int, int], Literal[0, 1]],
                 regret_fn: Callable[[int, int], float],
                 alpha: float = 2.50
                 ) -> Tuple[int, NDArray]:
    bandit = MultiSBM(K, T, alpha)
    regrets = []

    for t in range(T):
        i, j = bandit.select_next_pair(t)
        res = compare_fn(i, j)
        bandit.update_state(i, j, res, t)
        regret = regret_fn(i, j)
        regrets.append(regret)

    return bandit.get_best_arm(), np.cumsum(regrets)


class MultiSBM(DuelingBandit):

    def __init__(self, K: int, T: int, alpha: float = 2.50):
        self.K = K
        self.T = T
        self.sbm = [MAB(K, alpha) for _ in range(K)]
        self.current = random.randint(0, K - 1)


    def select_next_pair(self, *_) -> Tuple[int, int]:
        i = self.current
        j = self.sbm[i].select_next_arm()
        return i, j


    def update_state(self, i: int, j: int, res: Literal[0, 1], *_):
        if res == 1:
            self.sbm[i].update_state(j, 0.0)
        elif res == 0:
            self.sbm[i].update_state(j, 1.0)
        else:
            raise ValueError()
        self.current = j


    def get_best_arm(self) -> int:
        best_arms = [policy.get_best_arm() for policy in self.sbm]
        return max(set(best_arms), key=best_arms.count)
