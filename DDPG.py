import parl
from parl import layers
import paddle.fluid as fluid
import numpy as np

class ShareSubModel(parl.Model):
    def __init__(self, act_dim):
        self.conv1 = layers.conv2d(num_filters=16,filter_size=3,  act='relu')

        self.bn1= fluid.layers.batch_norm
        self.conv2 = layers.conv2d(num_filters=32,filter_size=3,  act='relu')

        self.bn2 = fluid.layers.batch_norm
        self.conv3 = layers.conv2d(num_filters=64,filter_size=3,  act='relu')

        self.bn3 = fluid.layers.batch_norm

        self.fc4 = layers.fc(size=act_dim, act='relu')
        self.fc5 = layers.fc(size=act_dim, act='relu')


    def forward(self, obs):
        hid1 = self.conv1(obs)
        hid1 = layers.pool2d(hid1,pool_size=2)
        hid1 = self.bn1(hid1)
        hid2 = self.conv2(hid1)
        hid2 = layers.pool2d(hid2,pool_size=2)
        hid2 = self.bn2(hid2)
        hid3 = self.conv3(hid2)
        hid3 = layers.pool2d(hid3,pool_size=2)
        hid3 = self.bn2(hid3)
        hid4 = self.fc4(hid3)
        flatten_obs=layers.flatten(obs, axis=1)
        concat = layers.concat([flatten_obs, hid4], axis=1)
        logits=self.fc5(concat)
        return logits

class ActorModel(parl.Model):

    def __init__(self, act_dim):
        self.fc5 = layers.fc(size=act_dim, act='softmax')

    def policy(self, hidden):
        logits=self.fc5(hidden)
        return logits


    # def __init__(self, act_dim):
    #     ######################################################################
    #     ######################################################################
    #     #
    #     # 2. 请配置model结构
    #     hid_size = 100
    #
    #     self.fc1 = layers.fc(size=hid_size, act='relu')
    #     self.fc3 = layers.fc(size=hid_size, act='relu')
    #     self.fc2 = layers.fc(size=act_dim, act='softmax')
    #     ######################################################################
    #     ######################################################################
    #
    # def policy(self, obs):
    #     ######################################################################
    #     ######################################################################
    #     #
    #     # 3. 请组装policy网络
    #     #
    #     hid = self.fc1(obs)
    #     hid = self.fc3(hid)
    #     logits = self.fc2(hid)
    #
    #     ######################################################################
    #     ######################################################################
    #     return logits


class CriticModel(parl.Model):
    def __init__(self):
        ######################################################################
        ######################################################################
        #
        # 4. 请配置model结构
        #
        hid_size = 100

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=1, act=None)
        ######################################################################
        ######################################################################

    def value(self, hidden, act):
        # 输入 state, action, 输出对应的Q(s,a)

        ######################################################################
        ######################################################################
        #
        # 5. 请组装Q网络
        #
        flatten_obs=layers.flatten(hidden, axis=1)
        concat = layers.concat([flatten_obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q2 = layers.squeeze(Q, axes=[1])
        return Q2

class Model(parl.Model):
    def __init__(self, act_dim):
        self.share_model = ShareSubModel(act_dim)
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy( self.share_model(obs))

    def value(self, obs, act):
        return self.critic_model.value(self.share_model(obs), act)

    def get_actor_params(self):
        return self.actor_model.parameters()+self.share_model.parameters()


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim=4):

        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        # 注意，在最开始的时候，先完全同步target_model和model的参数
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape= self.obs_dim , dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape= self.obs_dim , dtype='float32')
            act = layers.data(
                name='act', shape= [self.act_dim] , dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape= self.obs_dim , dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.actor_cost,self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)
    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
        # act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
        return act_prob

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        return act_prob

    def learn(self, obs, act_prob, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act_prob,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost