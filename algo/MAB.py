import numpy as np


class MAB():

    def __init__(self, K: int, alpha: float = 2.01):
        self.K = K
        self.alpha = alpha
        self.reset()


    def reset(self):
        self.Ni = np.zeros(self.K)  # Number of times each arm is selected
        self.Gi = np.zeros(self.K)  # Total reward for each arm

    
    def select_next_arm(self) -> int:
        n = np.sum(self.Ni)
        indices = np.zeros(self.K)

        for k in range(self.K):
            if self.Ni[k] == 0:
                return k
            indices[k] = (self.Gi[k] / self.Ni[k]) + np.sqrt((self.alpha * np.log(n)) / self.Ni[k])

        return np.argmax(indices)
    

    def update_state(self, k: int, r: float):
        self.Ni[k] += 1
        self.Gi[k] += r


    def get_best_arm(self) -> int:
        empirical_means = self.Gi / (self.Ni + 1e-6)
        return np.argmax(empirical_means)
