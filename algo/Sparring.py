from .utils import DuelingBandit
from .MAB import MAB

import numpy as np

from typing import Literal, Tuple, Callable
from numpy.typing import NDArray


def run_Sparring(K: int, T: int,
                 compare_fn: Callable[[int, int], Literal[0, 1]],
                 regret_fn: Callable[[int, int], float],
                 alpha: float = 2.50
                 ) -> Tuple[int, NDArray]:
    bandit = Sparring(K, T, alpha)
    regrets = []

    for t in range(T):
        i, j = bandit.select_next_pair(t)
        res = compare_fn(i, j)
        bandit.update_state(i, j, res, t)
        regret = regret_fn(i, j)
        regrets.append(regret)

    return bandit.get_best_arm(), np.cumsum(regrets)



class Sparring(DuelingBandit):

    def __init__(self, K: int, T: int, alpha=2.50):
        self.K = K
        self.T = T
        self.left = MAB(K, alpha)
        self.right = MAB(K, alpha)

    
    def select_next_pair(self, *_) -> Tuple[int, int]:
        i = self.left.select_next_arm()
        j = self.right.select_next_arm()
        return i, j
    

    def update_state(self, i: int, j: int, res: Literal[0, 1], *_):
        if res == 1:
            self.left.update_state(i, 1.)
            self.right.update_state(j, 0.)
        elif res == 0:
            self.left.update_state(i, 0.)
            self.right.update_state(j, 1.)
        else:
            raise ValueError()


    def get_best_arm(self) -> int:
        return self.left.get_best_arm()
