# prioritized experience replay (PER) buffer
# use the implementation from OpenAI's baselines
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        obs_t = np.asarray(obs_t, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        reward = np.float32(reward)
        obs_tp1 = np.asarray(obs_tp1, dtype=np.float32).reshape(-1)
        done = np.float32(done)

        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []

        for i in idxes:
            obs_t, action, reward, obs_tp1, done = self._storage[i]

            obses_t.append(np.asarray(obs_t, dtype=np.float32).reshape(-1))
            actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
            rewards.append(np.float32(reward))
            obses_tp1.append(np.asarray(obs_tp1, dtype=np.float32).reshape(-1))
            dones.append(np.float32(done))

        return (
            np.asarray(obses_t, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(obses_tp1, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


# ----------------------------------------------------------------------
# Segment tree helpers
# ----------------------------------------------------------------------
class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        elif start > mid:
            return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            return self._operation(
                self._reduce_helper(start, mid, 2 * node, node_start, mid),
                self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
            )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity - 1
        if end < 0:
            end += self._capacity
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=lambda x, y: x + y, neutral_element=0.0)

    def sum(self, start=0, end=None):
        return super().reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        idx = 1
        while idx < self._capacity:
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=min, neutral_element=float("inf"))

    def min(self, start=0, end=None):
        return super().reduce(start, end)


# ----------------------------------------------------------------------
# Prioritized replay buffer
# ----------------------------------------------------------------------
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super().__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        priority = self._max_priority ** self._alpha
        self._it_sum[idx] = priority
        self._it_min[idx] = priority

    def _sample_proportional(self, batch_size):
        res = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        for _ in range(batch_size):
            mass = random.random() * total
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.asarray(weights, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)

        for idx, priority in zip(idxes, priorities):
            priority = float(priority)
            assert priority > 0
            assert 0 <= idx < len(self._storage)

            value = priority ** self._alpha
            self._it_sum[idx] = value
            self._it_min[idx] = value

            self._max_priority = max(self._max_priority, priority)

    # Compatibility alias if old code calls camel-less version
    def updatepriorities(self, idxes, priorities):
        self.update_priorities(idxes, priorities)