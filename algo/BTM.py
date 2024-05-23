from .utils import DuelingBandit
from .MAB import MAB

import numpy as np
import random

from typing import Literal, Tuple, Callable
from numpy.typing import NDArray


def run_BTM(K: int, T: int,
            compare_fn: Callable[[int, int], Literal[0, 1]],
            regret_fn: Callable[[int, int], float],
            gamma: float = 1.2
            ) -> Tuple[int, NDArray]:
    bandit = BTM(K, T, gamma)
    regrets = []

    for t in range(T):
        i, j = bandit.select_next_pair(t)
        res = compare_fn(i, j)
        bandit.update_state(i, j, res, t)
        regret = regret_fn(i, j)
        regrets.append(regret)

    return bandit.get_best_arm(), np.cumsum(regrets)


class BTM(DuelingBandit):

    def __init__(self, K, T, gamma=1.2):
        self.K = K
        self.T = T
        self.gamma = gamma
        self.win_matrix = np.zeros((K, K), dtype=int)
        self.lose_matrix = np.zeros((K, K), dtype=int)
        self.W = set(range(K))
        self.n_b = np.zeros(K, dtype=int)
        self.w_b = np.zeros(K, dtype=int)

    
    def get_confidence_bound(self, k) -> float:
        if k == 0:
            return 1.0
        delta = 1.0 / (2.0 * self.T * self.K)
        return self.gamma * self.gamma * np.sqrt(np.log(1.0 / delta) / k)


    def minimum_trial_arm_and_num(self) -> Tuple[int, int]:
        b = None
        n_b_min = np.inf
        for i in self.W:
            if self.n_b[i] < n_b_min:
                b = i
                n_b_min = self.n_b[i]
        if b is None:
            raise ValueError()
        return b, n_b_min


    def minimum_trial_arm(self) -> int:
        return self.minimum_trial_arm_and_num()[0]


    def minimum_trial_num(self) -> int:
        return self.minimum_trial_arm_and_num()[1]
    

    def select_next_pair(self, *_) -> Tuple[int]:
        if not self.W:
            raise ValueError("No arm left.")
        elif len(self.W) == 1:  # Winner determined
            i = next(iter(self.W))
            return i, i
        else:  # Main routine
            b = self.minimum_trial_arm()
            self.W.remove(b)  # Avoid non-informative comparison
            b_prime = random.choice(list(self.W))
            self.W.add(b)
            return b, b_prime


    def update_state(self, i: int, j: int, res: Literal[0] | Literal[1], t: int):
        if len(self.W) == 1:  # In the exploration phase: no update
            return
        self.n_b[i] += 1
        if res == 1:
            self.win_matrix[i, j] += 1  # i beat j
            self.w_b[i] += 1
        elif res == 0:
            self.lose_matrix[i, j] += 1  # j beat i
        else: ValueError()

        # Remove bad arm
        P = {i: (self.w_b[i] / self.n_b[i] if self.n_b[i] > 0 else 0.5) for i in self.W}
        P_min_index = min(P, key=P.get)
        P_max_index = max(P, key=P.get)
        n_star = self.minimum_trial_num()
        c = self.get_confidence_bound(n_star)
        if P[P_min_index] + c < P[P_max_index] - c:
            b_prime = P_min_index
            self.W.remove(b_prime)
            for i in self.W:
                self.n_b[i] -= self.win_matrix[i, b_prime] + self.lose_matrix[i, b_prime]
                self.w_b[i] -= self.win_matrix[i, b_prime]


    def get_best_arm(self) -> int:
        empirical_means = self.w_b / np.maximum(self.n_b, 1)  # Avoid division by zero
        return np.argmax(empirical_means)
