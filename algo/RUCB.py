from .utils import DuelingBandit

import numpy as np

from typing import Tuple, Literal, Callable
from numpy.typing import NDArray


def run_RUCB(K: int, T: int,
             compare_fn: Callable[[int, int], Literal[0, 1]],
             regret_fn: Callable[[int, int], float],
             alpha: float = 0.51
             ) -> Tuple[int, NDArray]:
    bandit = RUCB(K, alpha)
    regrets = []

    for t in range(T):
        i, j = bandit.select_next_pair(t)
        res = compare_fn(i, j)
        bandit.update_state(i, j, res, t)
        regret = regret_fn(i, j)
        regrets.append(regret)

    return bandit.get_best_arm(), np.cumsum(regrets)


class RUCB(DuelingBandit):
    
    def __init__(self, K: int, alpha: float = 0.51):
        """
        Args:
            K (int): Number of arms.
            T (int): Horizon.
            alpha (float, optional): Exploration vs. exploitation tradeoff.
        """

        self.K = K
        self.alpha = alpha
        self.W = np.zeros((K, K), dtype=int)        # Number of times arm i beats arm j
        self.N = np.zeros((K, K), dtype=int)        # Number of comparisons between i and j; `W + W.T`.
        self.eps = 1e-6


    def get_condorcet_winner(self, U: NDArray):
        """Get a Condorcet winner.

        Args:
            U (NDArray): UCB matrix.
                U[i][j] represents the UCB value for the comparison between arms i and j.
        """
        cs = np.where(np.all(U >= 0.5, axis=1))[0]
        if len(cs) >= 1:
            # Condorcet winner exists. Randomly select one of the candidates.
            return np.random.choice(cs)
        else:
            # If no winner, select a random arm.
            return np.random.randint(0, self.K)
        

    def get_strongest_opponent(self, b: int, U: NDArray):
        """Get the strongest opponent for the chosen arm b.

        Args:
            b (int): Chosen arm.
            U (NDArray): UCB matrix, as above.

        Returns:
            int: Strongest opponent of b.
        """
        return np.argmax(U[:, b])
    

    def select_next_pair(self, t: int) -> Tuple[int, int]:
        U = np.zeros((self.K, self.K))
        mask = self.N == 0
        U[mask] = 1.0
        U[~mask] = self.W[~mask] / self.N[~mask] + np.sqrt(self.alpha * np.log(t + 1) / self.N[~mask])
        np.fill_diagonal(U, 0.5)
        b = self.get_condorcet_winner(U)
        op = self.get_strongest_opponent(b, U)
        return b, op
    

    def update_state(self, i: int, j: int, res: Literal[0, 1], *_):
        self.N[i, j] += 1
        self.N[j, i] += 1
        if res == 1: self.W[i, j] += 1
        elif res == 0: self.W[j, i] += 1
        else: raise ValueError()


    def get_best_arm(self) -> int:
        return np.argmax(np.sum(self.W, axis=1))
