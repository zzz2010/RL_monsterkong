
import parl
from parl import layers
import paddle.fluid as fluid
import numpy as np



class Model(parl.Model):
    def __init__(self, act_dim):
        num_layers = 18
        # ResNet
        # self.res = ResNet( depth=num_layers, num_classes=act_dim)

        # ResNet
        self.fc1 = layers.fc(size=200,  act='relu')
        self.fc2 = layers.fc(size=64,  act='relu')
        # self.fc2 = layers.conv2d(num_filters=32,filter_size=3,  act='relu')
        # self.fc3 = layers.conv2d(num_filters=64,filter_size=3,  act='relu')
        self.fc4 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        # hid3 = self.fc3(hid2)
        Q = self.fc4(hid2 )

        # Q = self.res(obs)
        return Q


class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=1e-6):

        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=self.obs_dim, dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=self.obs_dim, dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=self.obs_dim, dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数

        if sample < self.e_greed:
            act = np.random.rand(self.act_dim) #np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
            act =act/np.sum(act)
        else:
            act = self.predict(obs)  # 选择最优动作
            act+=1e-20-np.min(act)
            act = act / np.sum(act)

        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)

        return pred_Q

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        act = np.expand_dims(np.argmax(act,axis=1), -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

