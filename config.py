import random
import collections
import numpy as np
LEARN_FREQ = 1  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 60000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 64  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
rate_num = 1
LEARNING_RATE = 1e-6* rate_num  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
max_frames = 10000
e_greed=1

class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
               np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'), \
               np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

