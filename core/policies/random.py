import numpy as np

class RandomNormalPolicy:
    def __init__(
        self,
        action_dim,
        means: np.ndarray = np.array([0.1, 0.0]),
        stds: np.ndarray = np.array([0.1, np.pi / 8]),
        mins: np.ndarray = np.array([0.0, -np.pi / 4]),
        maxs: np.ndarray = np.array([1.0, np.pi / 4])
    ):
        self.action_dim = action_dim
        self.means = means
        self.stds = stds
        self.mins = mins
        self.maxs = maxs

    def __call__(
        self,
        obs
    ):
        metrics = {}
        action = np.clip(
            np.random.normal(loc=self.means, scale=self.stds),
            self.mins,
            self.maxs
        )

        return action, metrics