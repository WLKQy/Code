import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f

# BUFFER

import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

                Parameters
                ----------
                size: int
                    Max number of transitions to store in the buffer. When the buffer
                    overflows the old memories are dropped.
        """
        self._storage = []  # 储存生成数据的库
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, state, s_next, obs_t, action, reward, obs_tp1, done, padded):
        data = (state, s_next, obs_t, action, reward, obs_tp1, done, padded)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        states, s_nexts, obses_t, actions, rewards, obses_tp1, dones, paddeds = [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, s_next, obs_t, action, reward, obs_tp1, done, padded = data
            states.append(np.array(state))
            s_nexts.append(np.array(s_next))
            obses_t.append(np.array(obs_t))
            actions.append(np.array(action))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1))
            dones.append(done)
            paddeds.append(padded)
        return np.array(states), np.array(s_nexts), np.array(obses_t), np.array(actions), np.array(rewards), \
               np.array(obses_tp1), np.array(dones), np.array(paddeds)
        # return states, s_nexts, obses_t, actions, rewards, obses_tp1, dones

    def make_index(self, batch_size):  # batch_size=1024，生成随机的index
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]  # 从已有数据条中随机选择1024条数据

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
