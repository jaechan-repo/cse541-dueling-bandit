import numpy as np
import random

from typing import Literal, Callable, Tuple
from numpy.typing import NDArray


def run_IF1(K: int, T: int,
            compare_fn: Callable[[int, int], Literal[0, 1]],     
            regret_fn: Callable[[int, int], float]
            ) -> Tuple[float, NDArray]:
    delta = 1 / (T * (K ** 2))
    W = set(range(K))
    Res = np.zeros((K, K))

    def get_p_hat(i, j) -> float:
        if Res[i][j] + Res[j][i] == 0:
            return 0.5
        return Res[i][j] / (Res[i][j] + Res[j][i])
    
    def is_confident(p_hat, i, j, eps=1e-12) -> bool:
        t = Res[i][j] + Res[j][i] + eps
        c = np.sqrt(np.log(1 / delta) / t)
        return (0.5 < p_hat - c or 0.5 > p_hat + c)

    regrets = []
    t = 0

    b_hat = random.choice(tuple(W))
    W -= {b_hat}

    P_hat = {}

    while W and t < T:

        for b in W:
            res = compare_fn(b_hat, b)
            Res[b_hat][b] += res
            Res[b][b_hat] += 1 - res
            P_hat[b] = get_p_hat(b_hat, b)

            regret = regret_fn(b_hat, b)
            regrets.append(regret)
            t += 1

        remove_set = [b for b, p_hat in P_hat.items()
                      if p_hat > 0.5 and is_confident(p_hat, b_hat, b)]
        W -= set(remove_set)
        for b in remove_set: P_hat.pop(b)

        candidate_set = [b for b, p_hat in P_hat.items()
                         if p_hat < 0.5 and is_confident(p_hat, b_hat, b)]
        if candidate_set:
            b_hat = candidate_set[0]
            W -= {b_hat}
            P_hat = {}  # New round

    # Exploit!
    for _ in range(t, T):
        regret = regret_fn(b_hat, b_hat)
        regrets.append(regret)

    return b_hat, np.cumsum(regrets)
