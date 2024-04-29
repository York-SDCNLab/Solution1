import gym
import numpy as np

class ActionRewardResetWrapper(gym.Wrapper): # add artr to obs{}
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_size = env.action_size

    def step(self, action, metrics):
        obs, reward, done, info = self.env.step(action, metrics)
        obs["action"] = np.array(action, dtype=np.float32)
        obs["reward"] = np.array(reward, dtype=np.float32)
        obs["terminal"] = np.array(done, dtype=bool)
        obs["reset"] = np.array(False, dtype=bool)  # the env has been reset or not

        return obs, reward, done, info

    def reset(self):
        obs, reward, done, info = self.env.reset()
        obs["action"] = np.zeros((self.action_size,), dtype=np.float32)
        obs["reward"] = np.array(0.0, dtype=np.float32)
        obs["terminal"] = np.array(False, dtype=bool)
        obs["reset"] = np.array(True, dtype=bool)

        return obs, reward, done, info


class CollectWrapper(gym.Wrapper):  # create self.episode, info['episode'] and fill waypoints
    def __init__(self, env, paddings={}):
        super().__init__(env)
        self.env = env
        self.paddings = paddings

        self.episode = []  # key is step, compare with info['episode']

    def step(self, action, metrics):
        obs, reward, done, info = self.env.step(action, metrics)
        self.episode.append(obs.copy())  # copy obs dict as a item and add it to self.episode list
        if done:
            episode = {}  # episode dict can replace info, and just be returned, but we want more flexibility, so later we use episode as a value of the only one kv pair dict -- info
            for k in self.episode[0]:  # get all key of obs, so the 1st episode is enough
                data = []
                for t in self.episode:
                    if k in self.paddings:  # make sure all ep’s waypoints have same shape
                        shape, value = self.paddings[k]
                        data.append(np.pad(
                            t[k],
                            ((0, shape[0] - t[k].shape[0]), (0, 0)),
                            mode="constant",
                            constant_values=(value,)
                        ))
                    else:
                        data.append(t[k])

                episode[k] = np.array(data)

            info['episode'] = episode  # key is item (state waypoints image reward action terminal reset), compare with self.episode

        return obs, reward, done, info

    def reset(self):
        obs, reward, done, info = self.env.reset()
        self.episode = [obs.copy()]
        return obs, reward, done, info