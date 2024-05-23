from abc import ABC, abstractmethod
import numpy as np

from typing import Tuple, Literal
from numpy.typing import NDArray


def get_regret_fn(P: NDArray, best_arm: int = 0):
    def regret_fn(i: int, j: int) -> float:
        return P[best_arm][i] + P[best_arm][j] - 1
    return regret_fn


def get_compare_fn(P: NDArray):
    def compare_fn(i: int, j: int) -> Literal[0, 1]:
        return np.random.binomial(n=1, p=P[i][j], size=1)[0]
    return compare_fn


class DuelingBandit(ABC):

    @abstractmethod
    def select_next_pair(self, t: int) -> Tuple[int, int]:
        """Selects the next pair of arms.

        Args:
            t (int): Time step >= 0.

        Returns:
            Tuple[int, int]: Pair of arms.
        """
        pass

    @abstractmethod
    def update_state(self, i: int, j: int, res: Literal[0, 1], t: int):
        """Updates the bandit state based on the comparison result.

        Args:
            i (int): Arm 1.
            j (int): Arm 2.
            res (Literal[0, 1]): 1 if i beat j, 0 otherwise.
            t (int): Time step >= 0.
        """
        pass

    @abstractmethod
    def get_best_arm(self) -> int:
        pass
